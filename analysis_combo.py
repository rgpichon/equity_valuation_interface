"""
Analysis Combo Module
Combines all analysis results (DCF, Technical Analysis, Analyst Estimates, Fundamental Data)
into a unified database for the visual interface
"""

import pandas as pd
import numpy as np
import json
import pickle
import math
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from enhanced_comprehensive_data import EnhancedComprehensiveStockData
from analyst_estimates_data import AnalystEstimatesFetcher
from technical_analysis import TechnicalAnalysisEngine
from dcf_calculation import DCFCalculation

# ==================== INPUT NORMALIZATION HELPERS ====================

def _normalize_money_to_millions(x):
    """
    Return (value_millions, note).
    If x is None -> (np.nan, "none")
    If |x| >= 1e8 (looks like USD dollars) -> (x/1e6, "div1e6")
    Else -> (x, "as_is")  # already millions
    """
    if x is None:
        return (np.nan, "none")
    x = float(x)
    if abs(x) >= 1e8:
        return (x/1e6, "div1e6")
    return (x, "as_is")


def _pick_share_scale(shares_raw, market_cap_usd=None, spot_price_usd=None):
    """
    Decide whether shares are UNITS (x1), millions (x1e6), or billions (x1e9).
    We choose the scale whose implied price is closest to the spot price
    (if both market_cap and spot are usable). Otherwise fall back to heuristics.

    Returns (shares_units, scale_tag, implied_price)
    """
    if shares_raw is None or shares_raw <= 0:
        return (np.nan, "unknown", np.nan)
    s = float(shares_raw)

    # If already looks like UNITS for large caps
    if s >= 1e8:
        return (s, "x1", np.nan)

    candidates = [1.0, 1e6, 1e9]
    best = (s, "x1", np.nan)
    if market_cap_usd and market_cap_usd > 0 and spot_price_usd and 0.5 <= spot_price_usd <= 5000:
        best_err = float("inf")
        for c in candidates:
            implied = market_cap_usd / max(s*c, 1.0)
            # use log error so under/over scale penalize symmetrically
            err = abs(math.log(max(implied,1e-12)/spot_price_usd))
            if err < best_err:
                best_err = err
                best = (s*c, f"x{int(c)}" if c!=1.0 else "x1", implied)
        return best

    # No reliable price check; fall back by magnitude
    if s < 1e5:
        return (s*1e9, "x1000000000", np.nan)
    if s < 1e7:
        return (s*1e6, "x1000000", np.nan)
    return (s, "x1", np.nan)


def _normalize_for_dcf(scn: dict, facts: dict|None):
    """
    Normalize scenario fields for the DCF engine:
      - revenue_base (MILLIONS)
      - net_debt (MILLIONS)
      - shares (UNITS)

    facts may contain: {"market_cap_usd": float, "spot_price_usd": float}
    Returns (scn_norm: dict, unit_flags: dict)
    """
    scn_norm = dict(scn)  # shallow copy
    unit_flags = {}

    # Money → MILLIONS
    for k in ("revenue_base", "net_debt"):
        if k in scn_norm:
            val = scn_norm.get(k)
            val_m, note = _normalize_money_to_millions(val)
            scn_norm[k] = val_m
            unit_flags[f"{k}_note"] = note

    # Shares → UNITS via price-consistency
    if "shares" in scn_norm:
        mc = None if not facts else facts.get("market_cap_usd")
        px = None if not facts else facts.get("spot_price_usd")
        shares_units, scale_tag, implied = _pick_share_scale(scn_norm.get("shares"), mc, px)
        scn_norm["shares"] = shares_units
        unit_flags["shares_scale"] = scale_tag
        if not (implied is None or (isinstance(implied, float) and math.isnan(implied))):
            unit_flags["implied_price_check"] = float(implied)

    return scn_norm, unit_flags


# Robust numeric utilities
def _strip_to_scalar(x: Any) -> Any:
    if x is None: return None
    if isinstance(x, (np.ndarray,)): 
        for v in x[::-1]:
            s = _strip_to_scalar(v)
            if s is not None: return s
        return None
    if isinstance(x, (pd.Series, pd.Index)):
        for v in x[::-1]:
            s = _strip_to_scalar(v)
            if s is not None: return s
        return None
    if isinstance(x, (list, tuple)):
        for v in reversed(x):
            s = _strip_to_scalar(v)
            if s is not None: return s
        return None
    if isinstance(x, dict):
        preferred = ["value","raw","amount","total","latest","ttm","current","reported",
                     "revenue","sales","debt","total_debt","cash","cash_and_equivalents",
                     "shares","basic","diluted"]
        for k in preferred:
            if k in x:
                s = _strip_to_scalar(x[k])
                if s is not None: return s
        for v in x.values():
            s = _strip_to_scalar(v)
            if s is not None: return s
        return None
    return x

def to_num(x: Any, percent_ok: bool=False) -> Optional[float]:
    s = _strip_to_scalar(x)
    if s is None: return None
    if isinstance(s, (int, float)): 
        return float(s) if math.isfinite(float(s)) else None
    if isinstance(s, np.number):
        v = float(s); return v if math.isfinite(v) else None
    if isinstance(s, str):
        t = s.strip().replace(",", "")
        is_pct = t.endswith("%")
        if is_pct: t = t[:-1]
        try:
            v = float(t)
            if is_pct or (percent_ok and v > 1): v = v/100.0
            return v if math.isfinite(v) else None
        except Exception:
            return None
    return None

def safe_sub(a: Any, b: Any) -> float:
    A, B = to_num(a), to_num(b)
    if A is None and B is None: return 0.0
    if A is None: return -float(B)
    if B is None: return float(A)
    return float(A) - float(B)

def safe_div(a: Any, b: Any, default: float = float("nan")) -> float:
    A, B = to_num(a), to_num(b)
    if B in (None, 0): return default
    if A is None: return default
    return float(A)/float(B)


def load_company_fundamentals(ticker: str) -> dict:
    """
    Return floats for:
      shares_outstanding (count),
      total_debt,
      cash_and_equivalents,
      revenue_ttm,
      industry_text (str or None).
    Strategy:
      - Try enhanced_comprehensive_data / analyst_estimates_data if available.
      - Fallback to yfinance (if available in the project).
      - Coerce to floats; handle None.
      - Unit sanity:
          * if shares_outstanding and shares_outstanding < 1e4: shares_outstanding *= 1e6  # likely 'millions'
          * if revenue_ttm and revenue_ttm < 1e6: revenue_ttm *= 1e6                      # likely 'millions'
      - Return dict with None if missing (UI will guard).
    NOTE: No prints and no network beyond what the repo already supports. Pure getters only.
    """
    try:
        # Try to get data from existing data manager
        data_manager = EnhancedComprehensiveStockData.get_shared_instance(use_cache=True)
        
        # Get fundamental data
        fundamental_data = data_manager.export_fundamental_data(ticker)
        
        # Get OHLCV data to calculate shares outstanding from market cap and price
        ohlcv_data = data_manager.export_ohlcv_data(ticker)
        current_price = ohlcv_data['Close'].iloc[-1] if not ohlcv_data.empty else None
        
        # Calculate shares outstanding from market cap and price if available
        shares_outstanding = None
        market_cap = to_num(fundamental_data.get('market_cap'))
        enterprise_value = to_num(fundamental_data.get('enterprise_value'))
        
        if market_cap and current_price:
            shares_outstanding = market_cap / current_price
        elif enterprise_value and current_price:
            # Use enterprise value as proxy for market cap (rough approximation)
            estimated_market_cap = enterprise_value * 0.9  # Assume 10% net debt
            shares_outstanding = estimated_market_cap / current_price
            
        if shares_outstanding and shares_outstanding < 1e4:
            shares_outstanding *= 1e6  # likely 'millions'
        
        # Extract total debt - try different possible keys
        total_debt = (to_num(fundamental_data.get('total_debt')) or 
                     to_num(fundamental_data.get('debt')) or
                     to_num(fundamental_data.get('total_debt_to_equity')))
        
        # Extract cash and equivalents - try different possible keys
        cash_and_equivalents = (to_num(fundamental_data.get('cash_and_equivalents')) or 
                               to_num(fundamental_data.get('cash')) or
                               to_num(fundamental_data.get('cash_and_short_term_investments')))
        
        # Extract revenue TTM - try different possible keys
        revenue_ttm = (to_num(fundamental_data.get('revenue_ttm')) or 
                      to_num(fundamental_data.get('revenue')) or
                      to_num(fundamental_data.get('total_revenue')))
        
        # If no revenue data, estimate from enterprise value and PS ratio
        if not revenue_ttm and enterprise_value:
            ps_ratio = to_num(fundamental_data.get('ps_ratio'))
            if ps_ratio and ps_ratio > 0:
                # Revenue = Market Cap / PS Ratio
                estimated_market_cap = enterprise_value * 0.9  # Assume 10% net debt
                revenue_ttm = estimated_market_cap / ps_ratio
        
        if revenue_ttm and revenue_ttm < 1e6:
            revenue_ttm *= 1e6  # likely 'millions'
        
        # Try to get industry from company name or use default
        company_name = data_manager.get_company_name(ticker)
        industry_text = None
        if company_name:
            # Simple industry mapping based on company name
            name_lower = company_name.lower()
            if any(tech in name_lower for tech in ['apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla']):
                industry_text = "Technology"
            elif any(health in name_lower for health in ['pharma', 'medical', 'health', 'biotech']):
                industry_text = "Healthcare"
            elif any(finance in name_lower for finance in ['bank', 'financial', 'insurance']):
                industry_text = "Financial Services"
            else:
                industry_text = "Technology"  # Default
        
        return {
            "shares_outstanding": shares_outstanding,
            "total_debt": total_debt,
            "cash_and_equivalents": cash_and_equivalents,
            "revenue_ttm": revenue_ttm,
            "industry_text": industry_text
        }
        
    except Exception:
        # Fallback: return None values if data unavailable
        return {
            "shares_outstanding": None,
            "total_debt": None,
            "cash_and_equivalents": None,
            "revenue_ttm": None,
            "industry_text": None
        }


@dataclass
class StockAnalysis:
    """Complete analysis results for a single stock"""
    ticker: str
    company_name: str
    analysis_date: str
    
    # Price data
    current_price: float
    price_change_1d: float
    price_change_1w: float
    price_change_1m: float
    
    # Analyst estimates
    target_high: float
    target_median: float
    target_low: float
    analyst_rating: float
    number_of_analysts: int
    upside_to_target: float
    
    # Fundamental data
    market_cap: float
    pe_ratio: float
    forward_pe: float
    peg_ratio: float
    pb_ratio: float
    ps_ratio: float
    debt_to_equity: float
    roe: float
    revenue_growth: float
    earnings_growth: float
    dividend_yield: float
    beta: float
    
    # Technical analysis
    rsi: float
    macd_signal: str
    ma_trend: str
    bollinger_position: str
    stochastic_signal: str
    technical_signal: str
    technical_strength: int
    technical_confidence: float
    support_level: float
    resistance_level: float
    stop_loss: float
    target_price_technical: float
    
    # DCF valuation
    dcf_value: float
    dcf_upside: float
    wacc: float
    terminal_growth: float
    
    # Combined signals
    overall_signal: str
    signal_strength: int
    risk_level: str
    recommendation: str


class AnalysisCombo:
    """
    Comprehensive analysis combiner that aggregates all analysis modules
    """
    
    def __init__(self, use_cache: bool = True):
        print("🚀 Initializing Analysis Combo System...")
        
        # Initialize all analysis modules with shared data manager
        self.data_manager = EnhancedComprehensiveStockData.get_shared_instance(use_cache=use_cache)
        self.analyst_fetcher = AnalystEstimatesFetcher(data_manager=self.data_manager)
        self.technical_engine = TechnicalAnalysisEngine(data_manager=self.data_manager)
        self.dcf_calculator = DCFCalculation(data_manager=self.data_manager)
        
        # Storage for combined results
        self.combined_results = {}
        self.analysis_summary = None
        
        print("✅ Analysis Combo System initialized")
    
    def run_comprehensive_analysis(self, ticker: str) -> StockAnalysis:
        """
        Run all analysis modules for a single stock and combine results
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            StockAnalysis object with all combined results
        """
        print(f"\n📊 Running comprehensive analysis for {ticker}...")
        
        # Get company name
        company_name = self.data_manager.get_company_name(ticker)
        
        # 1. Get price data and calculate changes
        ohlcv_data = self.data_manager.export_ohlcv_data(ticker)
        current_price = ohlcv_data['Close'].iloc[-1]
        
        # Calculate price changes
        price_1d_ago = ohlcv_data['Close'].iloc[-2] if len(ohlcv_data) > 1 else current_price
        price_1w_ago = ohlcv_data['Close'].iloc[-6] if len(ohlcv_data) > 5 else current_price
        price_1m_ago = ohlcv_data['Close'].iloc[-22] if len(ohlcv_data) > 21 else current_price
        
        price_change_1d = ((current_price - price_1d_ago) / price_1d_ago) * 100
        price_change_1w = ((current_price - price_1w_ago) / price_1w_ago) * 100
        price_change_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
        
        # 2. Get analyst estimates
        analyst_data = self.analyst_fetcher.get_analyst_estimates(ticker)
        target_median = analyst_data['price_targets'].get('target_median', current_price)
        upside_to_target = ((target_median - current_price) / current_price) * 100 if target_median else 0
        
        # 3. Get fundamental data
        fundamental_data = self.data_manager.export_fundamental_data(ticker)
        
        # 4. Run technical analysis
        technical_results = self.technical_engine.analyze_stock(ticker)
        
        # 5. Run DCF valuation
        try:
            # Get real company data to determine industry
            company_data = self.dcf_calculator.fetch_company_data(ticker)
            industry = company_data.get('industry', 'Technology')
            sector = company_data.get('sector', 'Technology')
            
            # Get industry-specific parameters
            industry_params = self.dcf_calculator.calibrator.get_industry_assumptions(industry)
            terminal_growth = industry_params['long_term_growth']
            
            # Calculate independent revenue growth rates
            fundamental_revenue_growth = fundamental_data.get('revenue_growth', 0)
            
            # Use the independent growth calculation method
            revenue_growth_rates = self.dcf_calculator.calculate_independent_growth(
                industry=industry,
                current_revenue_growth=fundamental_revenue_growth,
                historical_growth=[0.05, 0.05, 0.05]  # Placeholder historical growth
            )
            
            print(f"✓ DCF using independent analysis for industry: {industry}")
            print(f"✓ Terminal growth: {terminal_growth:.1%}")
            
            dcf_results = self.dcf_calculator.run_dcf_from_ticker(
                ticker=ticker,
                revenue_growth_rates=revenue_growth_rates,
                terminal_growth_rate=terminal_growth,
                max_retries=1
            )
            
            dcf_value = dcf_results['summary']['value_per_share']
            dcf_upside = dcf_results['summary']['upside_downside'] * 100
            wacc = dcf_results['summary']['wacc']
        except Exception as e:
            print(f"⚠ DCF calculation failed: {e}")
            dcf_value = current_price
            dcf_upside = 0
            wacc = 0.10
            terminal_growth = 0.025
        
        # 6. Generate combined signals
        overall_signal, signal_strength, risk_level, recommendation = self._generate_combined_signals(
            technical_results=technical_results,
            dcf_upside=dcf_upside,
            analyst_upside=upside_to_target
        )
        
        # Create comprehensive analysis object
        analysis = StockAnalysis(
            ticker=ticker,
            company_name=company_name,
            analysis_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Price data
            current_price=round(current_price, 2),
            price_change_1d=round(price_change_1d, 2),
            price_change_1w=round(price_change_1w, 2),
            price_change_1m=round(price_change_1m, 2),
            
            # Analyst estimates
            target_high=analyst_data['price_targets'].get('target_high', current_price),
            target_median=target_median,
            target_low=analyst_data['price_targets'].get('target_low', current_price),
            analyst_rating=analyst_data['basic_info'].get('analyst_rating', 3.0),
            number_of_analysts=analyst_data['basic_info'].get('number_of_analysts', 0),
            upside_to_target=round(upside_to_target, 2),
            
            # Fundamental data
            market_cap=fundamental_data.get('market_cap', 0),
            pe_ratio=fundamental_data.get('pe_ratio'),
            forward_pe=fundamental_data.get('forward_pe'),
            peg_ratio=fundamental_data.get('peg_ratio'),
            pb_ratio=fundamental_data.get('pb_ratio'),
            ps_ratio=fundamental_data.get('ps_ratio'),
            debt_to_equity=fundamental_data.get('debt_to_equity'),
            roe=fundamental_data.get('roe'),
            revenue_growth=fundamental_data.get('revenue_growth'),
            earnings_growth=fundamental_data.get('earnings_growth'),
            dividend_yield=fundamental_data.get('dividend_yield'),
            beta=fundamental_data.get('beta', 1.0),
            
            # Technical analysis
            rsi=technical_results['rsi']['value'],
            macd_signal=technical_results['macd']['signal'],
            ma_trend=technical_results['moving_averages']['primary_trend'],
            bollinger_position=technical_results['bollinger_bands']['signal'],
            stochastic_signal=technical_results['stochastic']['signal'],
            technical_signal=technical_results['multi_indicator_signals']['overall_signal'],
            technical_strength=technical_results['multi_indicator_signals']['signal_strength'],
            technical_confidence=technical_results['multi_indicator_signals']['confidence'],
            support_level=technical_results['support_resistance'].get('nearest_support'),
            resistance_level=technical_results['support_resistance'].get('nearest_resistance'),
            stop_loss=technical_results['risk_management']['stop_loss_long'],
            target_price_technical=technical_results['risk_management']['target_price'],
            
            # DCF valuation
            dcf_value=round(dcf_value, 2),
            dcf_upside=round(dcf_upside, 2),
            wacc=round(wacc, 4),
            terminal_growth=round(terminal_growth, 4),
            
            # Combined signals
            overall_signal=overall_signal,
            signal_strength=signal_strength,
            risk_level=risk_level,
            recommendation=recommendation
        )
        
        # Store results
        self.combined_results[ticker] = analysis
        
        print(f"✅ Comprehensive analysis completed for {ticker}")
        return analysis
    
    def _generate_combined_signals(self, technical_results: Dict, dcf_upside: float, 
                                   analyst_upside: float) -> tuple:
        """
        Generate combined trading signals from all analysis sources
        
        Returns:
            (overall_signal, signal_strength, risk_level, recommendation)
        """
        signals = []
        weights = []
        
        # Technical signal (weight: 30%)
        tech_signal = technical_results['multi_indicator_signals']['overall_signal']
        tech_strength = technical_results['multi_indicator_signals']['signal_strength']
        
        if tech_signal == 'BUY':
            signals.append(tech_strength)
            weights.append(0.30)
        elif tech_signal == 'SELL':
            signals.append(-tech_strength)
            weights.append(0.30)
        else:
            signals.append(0)
            weights.append(0.30)
        
        # DCF signal (weight: 40%)
        if dcf_upside > 20:
            signals.append(8)
            weights.append(0.40)
        elif dcf_upside > 10:
            signals.append(6)
            weights.append(0.40)
        elif dcf_upside > -10:
            signals.append(0)
            weights.append(0.40)
        elif dcf_upside > -20:
            signals.append(-6)
            weights.append(0.40)
        else:
            signals.append(-8)
            weights.append(0.40)
        
        # Analyst signal (weight: 30%)
        if analyst_upside > 15:
            signals.append(8)
            weights.append(0.30)
        elif analyst_upside > 5:
            signals.append(6)
            weights.append(0.30)
        elif analyst_upside > -5:
            signals.append(0)
            weights.append(0.30)
        elif analyst_upside > -15:
            signals.append(-6)
            weights.append(0.30)
        else:
            signals.append(-8)
            weights.append(0.30)
        
        # Calculate weighted score
        weighted_score = sum(s * w for s, w in zip(signals, weights))
        
        # Determine overall signal
        if weighted_score > 3:
            overall_signal = 'STRONG BUY'
            signal_strength = min(10, int(5 + weighted_score))
        elif weighted_score > 1:
            overall_signal = 'BUY'
            signal_strength = min(10, int(5 + weighted_score))
        elif weighted_score > -1:
            overall_signal = 'HOLD'
            signal_strength = 5
        elif weighted_score > -3:
            overall_signal = 'SELL'
            signal_strength = max(1, int(5 + weighted_score))
        else:
            overall_signal = 'STRONG SELL'
            signal_strength = max(1, int(5 + weighted_score))
        
        # Determine risk level
        tech_risk = technical_results['multi_indicator_signals']['risk_level']
        if tech_risk == 'HIGH' or abs(dcf_upside) > 50:
            risk_level = 'HIGH'
        elif tech_risk == 'LOW' and abs(dcf_upside) < 20:
            risk_level = 'LOW'
        else:
            risk_level = 'MEDIUM'
        
        # Generate recommendation
        if overall_signal in ['STRONG BUY', 'BUY']:
            recommendation = f"Buy with {risk_level.lower()} risk"
        elif overall_signal == 'HOLD':
            recommendation = f"Hold position, monitor closely"
        else:
            recommendation = f"Consider selling, {risk_level.lower()} risk"
        
        return overall_signal, signal_strength, risk_level, recommendation
    
    def run_all_stocks(self) -> Dict[str, StockAnalysis]:
        """Run comprehensive analysis for all available stocks"""
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE ANALYSIS FOR ALL STOCKS")
        print("="*80)
        
        all_results = {}
        
        for ticker in self.data_manager.get_stock_list():
            try:
                analysis = self.run_comprehensive_analysis(ticker)
                all_results[ticker] = analysis
            except Exception as e:
                print(f"❌ Error analyzing {ticker}: {e}")
        
        # Create summary DataFrame
        self.analysis_summary = self.create_summary_dataframe(all_results)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE FOR ALL STOCKS")
        print("="*80)
        
        return all_results
    
    def create_summary_dataframe(self, results: Dict[str, StockAnalysis]) -> pd.DataFrame:
        """Create summary DataFrame from analysis results"""
        summary_data = []
        
        for ticker, analysis in results.items():
            summary_data.append({
                'Ticker': analysis.ticker,
                'Company': analysis.company_name,
                'Price': analysis.current_price,
                '1D %': analysis.price_change_1d,
                '1W %': analysis.price_change_1w,
                '1M %': analysis.price_change_1m,
                'Target': analysis.target_median,
                'Analyst ↑': analysis.upside_to_target,
                'DCF Value': analysis.dcf_value,
                'DCF ↑': analysis.dcf_upside,
                'Signal': analysis.overall_signal,
                'Strength': analysis.signal_strength,
                'Risk': analysis.risk_level,
                'Recommendation': analysis.recommendation
            })
        
        return pd.DataFrame(summary_data)
    
    def print_summary(self):
        """Print analysis summary"""
        if self.analysis_summary is None:
            print("No analysis data available. Run analysis first.")
            return
        
        print("\n" + "="*120)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*120)
        print(self.analysis_summary.to_string(index=False))
        print("="*120)
    
    def save_results(self, filepath: str = 'analysis_results.pkl'):
        """Save analysis results to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.combined_results, f)
        print(f"✅ Results saved to {filepath}")
    
    def load_results(self, filepath: str = 'analysis_results.pkl') -> Dict[str, StockAnalysis]:
        """Load analysis results from file"""
        with open(filepath, 'rb') as f:
            self.combined_results = pickle.load(f)
        print(f"✅ Results loaded from {filepath}")
        return self.combined_results
    
    def export_to_json(self, filepath: str = 'analysis_results.json'):
        """Export results to JSON format"""
        json_data = {}
        for ticker, analysis in self.combined_results.items():
            json_data[ticker] = asdict(analysis)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Results exported to {filepath}")
    
    def get_stock_analysis(self, ticker: str) -> Optional[StockAnalysis]:
        """Get analysis for a specific stock"""
        return self.combined_results.get(ticker)
    
    def run_scenario_analysis(self, ticker: str, scenario: dict) -> dict:
        """
        Run DCF analysis for a single scenario.
        
        Args:
            ticker: Stock ticker
            scenario: Scenario dict with inputs for project_operating_model and mode
            
        Returns:
            Dictionary with price_per_share, EV, wacc, g, exit_multiple (if applicable), projection_df
        """
        # Build inputs dict for project_operating_model (hardened with to_num)
        inputs = {
            'years': int(to_num(scenario.get('years', 5)) or 5),
            'revenue_base': to_num(scenario.get('revenue_base')) or 1000000000.0,
            'growth_path': [to_num(x) or 0.03 for x in scenario.get('growth_path', [0.03] * 5)],
            'ebitda_margin_now': to_num(scenario.get('ebitda_margin_now')) or 0.23,
            'margin_target': to_num(scenario.get('margin_target')) or 0.25,
            'tax_rate': to_num(scenario.get('tax_rate'), percent_ok=True) or 0.22,
            'capex_pct_sales': to_num(scenario.get('capex_pct_sales'), percent_ok=True) or 0.05,
            'nwc_pct_sales': to_num(scenario.get('nwc_pct_sales'), percent_ok=True) or 0.03,
            'depr_pct_sales': to_num(scenario.get('depr_pct_sales'), percent_ok=True) or 0.03,
            'shares': to_num(scenario.get('shares')) or 1000000000.0,
            'net_debt': to_num(scenario.get('net_debt')) or 0.0
        }
        
        # Project operating model
        projection_df = self.dcf_calculator.project_operating_model(inputs, {})
        
        # Build adjustments for valuation (hardened)
        adjustments = {
            'shares_outstanding': to_num(scenario.get('shares')) or 1000000000.0,
            'net_debt': to_num(scenario.get('net_debt')) or 0.0
        }
        
        # Get WACC and terminal growth from scenario (hardened)
        wacc = to_num(scenario.get('wacc')) or 0.10
        terminal_g = to_num(scenario.get('terminal_g')) or 0.025
        
        # Run valuation based on mode
        mode = scenario.get('mode', 'perpetuity')
        
        if mode == 'perpetuity':
            valuation = self.dcf_calculator.value_with_perpetuity(
                fcfs_df=projection_df,
                wacc=wacc,
                terminal_g=terminal_g,
                adjustments=adjustments
            )
            exit_multiple = None
        else:  # exit mode
            exit_multiple = to_num(scenario.get('exit_multiple')) or 10.0
            valuation = self.dcf_calculator.value_with_exit_multiple(
                fcfs_df=projection_df,
                wacc=wacc,
                exit_multiple=exit_multiple,
                base_metric="EBITDA",
                adjustments=adjustments
            )
        
        return {
            'price_per_share': valuation['price_per_share'],
            'EV': valuation['enterprise_value'],
            'wacc': wacc,
            'g': terminal_g,
            'exit_multiple': exit_multiple,
            'projection_df': projection_df
        }
    
    def run_sensitivity(self, projection_df: pd.DataFrame, wacc: float, g: float) -> pd.DataFrame:
        """
        Run sensitivity analysis varying WACC and terminal growth around base values.
        
        Args:
            projection_df: Projected cash flows DataFrame
            wacc: Base WACC for range calculation
            g: Base terminal growth for range calculation
            
        Returns:
            Sensitivity matrix DataFrame
        """
        # Build adjustments (minimal - just for shares and net_debt)
        adjustments = {
            'shares_outstanding': 1000000000.0,  # Default shares
            'net_debt': 0.0  # Default net debt
        }
        
        # Build WACC and g ranges (±200/±100 bps around base) - hardened
        wacc_safe = to_num(wacc) or 0.10
        g_safe = to_num(g) or 0.025
        wacc_values = [wacc_safe - 0.02, wacc_safe - 0.01, wacc_safe, wacc_safe + 0.01, wacc_safe + 0.02]
        g_values = [g_safe - 0.01, g_safe - 0.005, g_safe, g_safe + 0.005, g_safe + 0.01]
        
        # Build sensitivity matrix
        sensitivity_df = self.dcf_calculator.build_sensitivity_matrix(
            fcfs_df=projection_df,
            wacc_values=wacc_values,
            g_values=g_values,
            adjustments=adjustments
        )
        
        return sensitivity_df


def main():
    """Main function to demonstrate the analysis combo system"""
    
    # Initialize combo system
    combo = AnalysisCombo(use_cache=True)
    
    # Run analysis for all stocks
    results = combo.run_all_stocks()
    
    # Print summary
    combo.print_summary()
    
    # Save results for interface
    combo.save_results('analysis_results.pkl')
    combo.export_to_json('analysis_results.json')
    
    return combo


# Calibration picker functions
def pick_risk_anchors(country: str = None) -> dict:
    """RISK_DEFAULTS["erp_anchor"] is a dict; choose a single numeric, default to "developed" else first value."""
    import dcf_calibration as calib
    erp_map = calib.RISK_DEFAULTS.get("erp_anchor", {})
    erp = (erp_map.get("developed") if isinstance(erp_map, dict) else calib.RISK_DEFAULTS.get("erp_anchor", 0.05)) or 0.05
    rf = to_num(calib.RISK_DEFAULTS.get("risk_free_anchor", 0.04)) or 0.04
    return {"risk_free": float(rf), "erp": float(erp)}

def pick_industry_defaults(industry_text: str = None) -> tuple:
    """Map by substring lowercased; else return first key/value from INDUSTRY_DEFAULTS."""
    import dcf_calibration as calib
    d = calib.INDUSTRY_DEFAULTS
    if not isinstance(d, dict) or not d:
        raise ValueError("INDUSTRY_DEFAULTS missing or not a dict")
    if industry_text:
        t = industry_text.lower()
        for k in d.keys():
            if k.lower() in t or t in k.lower():
                return k, d[k]
    # fallback
    k0 = next(iter(d.keys()))
    return k0, d[k0]


# Thin dict-based helpers for UI
def run_scenario_analysis(ticker: str, scenario: dict, industry_hint: str = None) -> dict:
    """
    Finance-grade DCF adapter for UI consumption.
    Coerces and validates scenario values, calls pure DCF functions.
    
    Args:
        ticker: Stock ticker symbol
        scenario: Dictionary with DCF parameters
        industry_hint: Optional industry hint for defaults
        
    Returns:
        Dict with projection_df, valuation, breakdown, wacc, g, exit_multiple
        
    Raises:
        ValueError: If required numeric parameters are None or invalid
    """
    import dcf_calculation as dc
    import math
    
    # Get company facts for normalization
    try:
        combo = AnalysisCombo()
        facts = combo.get_company_fundamentals(ticker)
    except:
        facts = None
    
    # Build facts_ext for normalization
    facts_ext = {
        "market_cap_usd": facts.get("market_cap") if facts else None,
        "spot_price_usd": facts.get("current_price") if facts else None
    }
    
    # Normalize inputs for DCF engine
    scn_norm, unit_flags = _normalize_for_dcf(scenario, facts_ext)
    
    # Create DCF calculation instance
    dcf = dc.DCFCalculation()
    
    # Run projection model with normalized inputs
    projection_df = dcf.project_operating_model(scn_norm, {})
    
    # Build adjustments for valuation
    adjustments = {
        'net_debt': scn_norm['net_debt'],
        'shares': scn_norm['shares']
    }
    
    # Determine valuation mode and call appropriate function
    mode = scn_norm.get("mode", "perpetuity")
    
    if mode == "blend":
        # Use blend mode: average of perpetuity and exit-multiple
        valuation = dcf.value_with_blend(
            projection_df, 
            scn_norm['wacc'], 
            scn_norm['g'], 
            scn_norm['exit_multiple'], 
            'EBITDA', 
            adjustments, 
            midyear=True
        )
    elif mode == "exit":
        # Use exit multiple valuation
        valuation = dcf.value_with_exit_multiple(
            projection_df, 
            scn_norm['wacc'], 
            scn_norm['exit_multiple'], 
            'EBITDA', 
            adjustments, 
            midyear=True
        )
    else:
        # Use perpetuity valuation (default)
        valuation = dcf.value_with_perpetuity(
            projection_df, 
            scn_norm['wacc'], 
            scn_norm['g'], 
            adjustments, 
            midyear=True
        )
    
    # Get industry defaults for breakdown
    industry_defaults = None
    if industry_hint:
        try:
            industry_name, industry_params = get_industry_defaults(industry_hint)
            industry_defaults = {
                "industry_hint": industry_hint,
                "industry_name": industry_name,
                "default_wacc": 0.085,  # Default WACC
                "default_terminal_g": industry_params.get("lt_growth_cap", 0.025),
                "default_exit_multiple": industry_params.get("exit_multiple_hint", 12.0),
                "margin_target": industry_params.get("margin_target", 0.30),
                "capex_pct_sales": industry_params.get("capex_pct_sales", 0.04),
                "nwc_pct_sales": industry_params.get("nwc_pct_sales", 0.02),
                "tax_rate": industry_params.get("tax_rate", 0.17)
            }
        except Exception:
            industry_defaults = None
    
    # Add industry defaults to breakdown if available
    if industry_defaults and 'breakdown' in valuation:
        valuation['breakdown']['industry_defaults'] = industry_defaults
    
    # Add simple sanity flags into valuation["breakdown"]
    if 'breakdown' in valuation:
        breakdown = valuation['breakdown']
        
        # Calculate implied EV/EBITDA multiple for peer comparison
        try:
            final_ebitda = float(projection_df.iloc[-1]['EBITDA'])
            implied_ev_ebitda = valuation['EV'] / max(final_ebitda, 1e-9)
            
            if implied_ev_ebitda < 5 or implied_ev_ebitda > 30:
                breakdown['peer_flag'] = "out_of_band"
                breakdown['implied_EV_EBITDA'] = implied_ev_ebitda
        except (IndexError, KeyError, ZeroDivisionError):
            pass  # Skip sanity check if data unavailable
    
    # Return richer result with deterministic diagnostics
    return {
        "projection_df": projection_df,
        "valuation": valuation,
        "assumptions": {
            "wacc": float(scn_norm["wacc"]),
            "g": float(scn_norm["g"]) if scn_norm.get("g") is not None else None,
            "exit_multiple": float(scn_norm["exit_multiple"]) if scn_norm.get("exit_multiple") is not None else None,
            "midyear": True,
            "unit_flags": unit_flags,
            "shares_used": scn_norm["shares"],
            "revenue_base_m": scn_norm["revenue_base"],
            "net_debt_m": scn_norm["net_debt"],
            "inputs_used": {
                "revenue_base": float(scn_norm["revenue_base"]),
                "ebitda_margin_now": float(scn_norm["ebitda_margin_now"]),
                "margin_target": float(scn_norm["margin_target"]),
                "capex_pct_sales": float(scn_norm["capex_pct_sales"]),
                "nwc_pct_sales": float(scn_norm["nwc_pct_sales"]),
                "depr_pct_sales": float(scn_norm["depr_pct_sales"]),
                "tax_rate": float(scn_norm["tax_rate"]),
                "shares": float(scn_norm["shares"]),
                "net_debt": float(scn_norm["net_debt"])
            }
        }
    }

def run_sensitivity(projection_df: pd.DataFrame, wacc: float, g: float) -> pd.DataFrame:
    """
    Finance-grade sensitivity analysis adapter.
    Builds ±200 bps around wacc and ±100 bps around g (5×5).
    
    Args:
        projection_df: DataFrame with DCF projections
        wacc: Base WACC value
        g: Base terminal growth rate
        
    Returns:
        DataFrame with sensitivity matrix
    """
    import dcf_calculation as dc
    
    # Build ±200 bps around wacc and ±100 bps around g (5×5)
    wacc_values = [wacc-0.02, wacc-0.01, wacc, wacc+0.01, wacc+0.02]
    g_values = [g-0.01, g-0.005, g, g+0.005, g+0.01]
    
    # Build adjustments (minimal - just for shares and net_debt)
    adjustments = {
        'shares': 1000000000.0,  # Default shares
        'net_debt': 0.0  # Default net debt
    }
    
    return dc.build_sensitivity_matrix(
        projection_df, 
        wacc_values=wacc_values, 
        g_values=g_values, 
        adjustments=adjustments
    )


def get_industry_defaults(industry_hint: str|None = None) -> dict:
    """
    Get industry-specific WACC and terminal growth defaults.
    
    Args:
        industry_hint: Industry hint (e.g., 'Technology', 'Healthcare', 'Financial')
        
    Returns:
        Dict with 'wacc', 'terminal_g', 'exit_multiple' defaults
    """
    if not industry_hint:
        return {"wacc": 0.085, "terminal_g": 0.025, "exit_multiple": 12.0}
    
    industry_hint_lower = industry_hint.lower()
    
    # Industry-specific defaults
    industry_defaults = {
        # Technology
        "tech": {"wacc": 0.090, "terminal_g": 0.030, "exit_multiple": 15.0},
        "technology": {"wacc": 0.090, "terminal_g": 0.030, "exit_multiple": 15.0},
        "software": {"wacc": 0.095, "terminal_g": 0.035, "exit_multiple": 18.0},
        
        # Healthcare
        "healthcare": {"wacc": 0.080, "terminal_g": 0.025, "exit_multiple": 14.0},
        "pharma": {"wacc": 0.075, "terminal_g": 0.020, "exit_multiple": 12.0},
        "biotech": {"wacc": 0.085, "terminal_g": 0.025, "exit_multiple": 16.0},
        
        # Financial
        "financial": {"wacc": 0.080, "terminal_g": 0.020, "exit_multiple": 10.0},
        "banking": {"wacc": 0.075, "terminal_g": 0.015, "exit_multiple": 8.0},
        "insurance": {"wacc": 0.070, "terminal_g": 0.020, "exit_multiple": 12.0},
        
        # Consumer
        "consumer": {"wacc": 0.085, "terminal_g": 0.025, "exit_multiple": 12.0},
        "retail": {"wacc": 0.090, "terminal_g": 0.020, "exit_multiple": 10.0},
        "food": {"wacc": 0.075, "terminal_g": 0.020, "exit_multiple": 12.0},
        
        # Industrial
        "industrial": {"wacc": 0.085, "terminal_g": 0.025, "exit_multiple": 12.0},
        "manufacturing": {"wacc": 0.080, "terminal_g": 0.020, "exit_multiple": 10.0},
        "automotive": {"wacc": 0.085, "terminal_g": 0.025, "exit_multiple": 8.0},
        
        # Energy
        "energy": {"wacc": 0.090, "terminal_g": 0.025, "exit_multiple": 8.0},
        "oil": {"wacc": 0.095, "terminal_g": 0.020, "exit_multiple": 6.0},
        "utilities": {"wacc": 0.070, "terminal_g": 0.025, "exit_multiple": 15.0},
        
        # Default
        "default": {"wacc": 0.085, "terminal_g": 0.025, "exit_multiple": 12.0}
    }
    
    # Find best match
    for key, defaults in industry_defaults.items():
        if key in industry_hint_lower:
            return defaults
    
    return industry_defaults["default"]


def scenario_presets(industry_hint: str|None = None) -> dict:
    """
    Returns a dict with three independent scenario dicts: 'Bear','Base','Bull'.
    Each dict contains only numeric fields expected by run_scenario_analysis.
    Uses industry-specific defaults for WACC and terminal growth.
    
    Args:
        industry_hint: Industry hint for WACC/g defaults (e.g., 'Technology', 'Healthcare')
        
    Returns:
        Dict with 'Bear', 'Base', 'Bull' keys, each containing scenario dict
        
    Variations:
        Bear: wacc +200bps, g -50bps, margin_target -300bps, growth_path ~1.5%
        Base: industry-specific defaults
        Bull: wacc -150bps, g +50bps, margin_target +200bps, growth_path ~4.5%
    """
    import copy
    
    # Get industry-specific defaults
    industry_defaults = get_industry_defaults(industry_hint)
    base_wacc = industry_defaults["wacc"]
    base_terminal_g = industry_defaults["terminal_g"]
    exit_multiple = industry_defaults["exit_multiple"]
    
    # Base scenario (industry-specific defaults)
    base_scenario = {
        "years": 5,
        "revenue_base": 1_000_000_000.0,  # Will be overridden by fundamentals
        "growth_path": [0.03] * 5,  # 3% CAGR
        "ebitda_margin_now": 0.30,
        "margin_target": 0.33,
        "tax_rate": 0.17,
        "capex_pct_sales": 0.04,
        "nwc_pct_sales": 0.02,
        "depr_pct_sales": 0.03,
        "shares": 1_000_000_000.0,  # Will be overridden by fundamentals
        "net_debt": 0.0,  # Will be overridden by fundamentals
        "wacc": base_wacc,
        "g": base_terminal_g,
        "exit_multiple": exit_multiple,
        "mode": "dual_terminal"  # Use dual terminal valuation
    }
    
    # Bear scenario (pessimistic)
    bear_scenario = copy.deepcopy(base_scenario)
    bear_scenario.update({
        "wacc": base_wacc + 0.02,  # +200bps
        "g": max(0.01, base_terminal_g - 0.005),  # -50bps, min 1%
        "margin_target": 0.30,  # -300bps = 30%
        "growth_path": [0.015] * 5,  # 1.5% CAGR
        "exit_multiple": exit_multiple * 0.8,  # 20% lower exit multiple
    })
    
    # Ensure wacc > g for bear scenario
    if bear_scenario["wacc"] <= bear_scenario["g"]:
        bear_scenario["g"] = bear_scenario["wacc"] - 0.0025  # Adjust g down by 25bps
    
    # Bull scenario (optimistic)
    bull_scenario = copy.deepcopy(base_scenario)
    bull_scenario.update({
        "wacc": max(0.05, base_wacc - 0.015),  # -150bps, min 5%
        "g": min(0.05, base_terminal_g + 0.005),  # +50bps, max 5%
        "margin_target": 0.35,  # +200bps = 35%
        "growth_path": [0.045] * 5,  # 4.5% CAGR
        "exit_multiple": exit_multiple * 1.2,  # 20% higher exit multiple
    })
    
    # Ensure wacc > g for bull scenario
    if bull_scenario["wacc"] <= bull_scenario["g"]:
        bull_scenario["g"] = bull_scenario["wacc"] - 0.0025  # Adjust g down by 25bps
    
    return {
        "Bear": bear_scenario,
        "Base": base_scenario,
        "Bull": bull_scenario
    }


def calculate_dcf_range(results: dict) -> dict:
    """
    Calculate DCF range from multiple scenario results.
    
    Args:
        results: Dict with scenario results from run_multiple_scenarios
        
    Returns:
        Dict with min/max/range information
    """
    prices = []
    evs = []
    
    for scenario_name, result in results.items():
        if "valuation" in result:
            prices.append(result["valuation"]["price_per_share"])
            evs.append(result["valuation"]["EV"])
    
    if not prices:
        return {"min_price": 0, "max_price": 0, "range": 0, "min_ev": 0, "max_ev": 0}
    
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    min_ev = min(evs)
    max_ev = max(evs)
    
    return {
        "min_price": float(min_price),
        "max_price": float(max_price),
        "range": float(price_range),
        "min_ev": float(min_ev),
        "max_ev": float(max_ev),
        "price_range_pct": float((max_price - min_price) / min_price * 100) if min_price > 0 else 0
    }


def run_multiple_scenarios(ticker: str, presets: dict, industry_hint: str = None) -> dict:
    """
    For each key in presets, call run_scenario_analysis(ticker, presets[name]) and 
    return a dict {name: {"valuation":..., "projection_df":..., "wacc":..., "g":...}}
    
    Args:
        ticker: Stock ticker symbol
        presets: Dict with scenario names as keys and scenario dicts as values
        industry_hint: Optional industry hint for defaults
        
    Returns:
        Dict with scenario names as keys and analysis results as values
    """
    results = {}
    
    for name, scenario in presets.items():
        try:
            result = run_scenario_analysis(ticker, scenario, industry_hint)
            results[name] = {
                "valuation": result["valuation"],
                "projection_df": result["projection_df"],
                "wacc": result["wacc"],
                "g": result["g"],
                "breakdown": result.get("breakdown", {})
            }
        except Exception as e:
            # If scenario fails, create a minimal result
            results[name] = {
                "valuation": {"EV": 0, "equity_value": 0, "price_per_share": 0},
                "projection_df": pd.DataFrame(),
                "wacc": scenario.get("wacc", 0),
                "g": scenario.get("g", 0),
                "breakdown": {"error": str(e)}
            }
    
    # Calculate DCF range
    dcf_range = calculate_dcf_range(results)
    results["_dcf_range"] = dcf_range
    
    return results


# Export list for module
__all__ = [
    'AnalysisCombo', 
    'StockAnalysis', 
    'load_company_fundamentals',
    'run_scenario_analysis', 
    'run_sensitivity',
    'scenario_presets',
    'run_multiple_scenarios',
    'get_industry_defaults',
    'calculate_dcf_range',
    'pick_risk_anchors', 
    'pick_industry_defaults'
]

if __name__ == "__main__":
    analysis_combo = main()


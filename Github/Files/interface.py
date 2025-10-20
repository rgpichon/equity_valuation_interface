"""
Interactive Stock Analysis Interface
Visual dashboard for comprehensive stock analysis using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
import copy
import math
from datetime import datetime
from typing import Dict, Optional, Any, Iterable
import warnings
warnings.filterwarnings('ignore')

# Import our analysis modules
from analysis_combo import AnalysisCombo, StockAnalysis, load_company_fundamentals #type: ignore
from enhanced_comprehensive_data import EnhancedComprehensiveStockData

# Robust imports with fallbacks
try:
    from analysis_combo import run_scenario_analysis as _run_scenario_analysis
except Exception:
    _run_scenario_analysis = None

try:
    from analysis_combo import run_sensitivity as _run_sensitivity
except Exception:
    _run_sensitivity = None

# UI only uses the adapter functions from analysis_combo.py

def _scalar(x: Any) -> Any:
    """Extract a usable numeric scalar from nested or structured inputs."""
    if x is None: return None
    if isinstance(x, (np.ndarray, list, tuple, pd.Series, pd.Index)):
        for v in x:
            s = _scalar(v)
            if s is not None: return s
        return None
    if isinstance(x, dict):
        for k in ["value","raw","amount","total","latest","ttm","current",
                  "revenue","sales","debt","total_debt","cash","cash_and_equivalents",
                  "shares","basic","diluted"]:
            if k in x:
                s = _scalar(x[k])
                if s is not None: return s
        for v in x.values():
            s = _scalar(v)
            if s is not None: return s
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def to_num(x: Any, percent_ok=False) -> Optional[float]:
    """Convert arbitrary input to a numeric float or None."""
    s = _scalar(x)
    if s is None: return None
    if isinstance(s, str):
        t = s.strip().replace(",", "")
        if t.endswith("%"):
            try: return float(t[:-1]) / 100.0
            except: return None
        try: return float(t)
        except: return None
    return float(s)

def fsub(a, b): 
    A, B = to_num(a), to_num(b)
    if A is None and B is None: return 0.0
    if A is None: return -B
    if B is None: return A
    return A - B

def fdiv(a, b, default=float("nan")):
    A, B = to_num(a), to_num(b)
    if A is None or B is None or B == 0: return default
    return A / B

def get_num(state, key: str, default=None, percent_ok=False):
    try: val = state.get(key, default)
    except Exception: val = default
    return to_num(val, percent_ok=percent_ok) or default

def _normalize_dcf_result(obj: Any) -> dict:
    """
    Returns a canonical dict structure. Never raises.
    """
    res = {
        "projection_df": None,
        "valuation": {"EV": 0.0, "equity_value": 0.0, "price_per_share": 0.0},
        "breakdown": {},
        "wacc": None,
        "g": None,
        "exit_multiple": None,
    }
    
    if obj is None:
        return res
    
    # Dict path
    if isinstance(obj, dict):
        res.update({k: obj.get(k) for k in res.keys()})
        # Accept alternative keys for valuation
        if not res['valuation']:
            for alt in ('value', 'values', 'valuation_result'):
                if isinstance(obj.get(alt), dict):
                    res['valuation'] = obj[alt]
                    break
        return res
    
    # Object path - try common attributes
    try:
        if hasattr(obj, 'projection_df'):
            res['projection_df'] = getattr(obj, 'projection_df')
        elif hasattr(obj, 'projection') and isinstance(getattr(obj, 'projection'), pd.DataFrame):
            res['projection_df'] = getattr(obj, 'projection')
        
        if hasattr(obj, 'valuation'):
            val = getattr(obj, 'valuation')
            if isinstance(val, dict):
                res['valuation'] = val
            else:
                # Pull common fields as attributes
                ev = getattr(val, 'EV', None) or getattr(val, 'ev', None)
                eq = getattr(val, 'equity_value', None) or getattr(val, 'equity', None)
                pps = getattr(val, 'price_per_share', None) or getattr(val, 'pps', None)
                part = {}
                if ev is not None: part['EV'] = float(ev)
                if eq is not None: part['equity_value'] = float(eq)
                if pps is not None: part['price_per_share'] = float(pps)
                res['valuation'] = part
        
        # Optional breakdown / params
        for k in ('breakdown', 'wacc', 'g', 'exit_multiple'):
            if hasattr(obj, k):
                res[k] = getattr(obj, k)
    except Exception:
        pass
    
    return res

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Global UI Density Toggle (optional but recommended) ----
with st.sidebar:
    density = st.selectbox(
        "UI density",
        options=["Normal", "Compact"],
        index=1  # default to Compact
    )

_is_compact = (density == "Compact")

# ---- Global Style Overrides ----
_base_font = "0.90rem" if _is_compact else "1.0rem"
_table_font = "0.85rem" if _is_compact else "0.95rem"
_head_scale = "90%" if _is_compact else "100%"
_metric_font = "0.90rem" if _is_compact else "1.0rem"
_caption_font = "0.85rem" if _is_compact else "0.95rem"
_line_height = "1.25rem" if _is_compact else "1.45rem"
_pad_top = "1rem" if _is_compact else "1.5rem"
_pad_bottom = "0.2rem" if _is_compact else "0.8rem"

st.markdown(f"""
<style>

/* 1) Reduce global font size slightly */
html, body, [class*="css"] {{
  font-size: {_base_font} !important;
}}

/* 2) Headings slightly smaller but still distinct */
h1, h2, h3, h4 {{
  font-size: {_head_scale} !important;
}}

/* 3) Compact main container padding */
.block-container {{
  padding-top: {_pad_top} !important;
  padding-bottom: {_pad_bottom} !important;
}}

/* 4) Metric cards */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
  font-size: {_metric_font} !important;
  line-height: {_line_height} !important;
}}
/* Tighten metric spacing */
[data-testid="stMetricValue"] div {{
  margin-top: 0.1rem !important;
}}

/* 5) DataFrame/table font sizes */
.stDataFrame table, .stDataFrame th, .stDataFrame td {{
  font-size: {_table_font} !important;
  line-height: 1.2rem !important;
}}
/* Narrow header padding */
.stDataFrame th {{
  padding-top: 0.35rem !important;
  padding-bottom: 0.35rem !important;
}}

/* 6) Expander headers and content */
.streamlit-expanderHeader {{
  font-size: 0.95rem !important;
}}
.streamlit-expanderContent p, 
.streamlit-expanderContent li, 
.streamlit-expanderContent span {{
  font-size: {_caption_font} !important;
  line-height: {_line_height} !important;
}}

/* 7) General paragraph/list text */
p, li, span {{
  font-size: {_base_font} !important;
  line-height: {_line_height} !important;
}}

/* 8) Reduce vertical gaps between elements without breaking layout */
.css-1y4p8pa, .css-1dp5vir, .css-1mi714g {{
  margin-top: 0.25rem !important;
  margin-bottom: 0.25rem !important;
}}

/* 9) Keep charts readable: do NOT alter Plotly canvas font sizes directly */

/* 10) Original custom CSS classes (preserved) */
.metric-card {{
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}}

.buy-signal {{
    border-left-color: #2ca02c !important;
}}

.sell-signal {{
    border-left-color: #d62728 !important;
}}

.hold-signal {{
    border-left-color: #ff7f0e !important;
}}

.big-number {{
    font-size: 2rem;
    font-weight: bold;
    margin: 0;
}}

.small-text {{
    font-size: 0.8rem;
    color: #666;
    margin: 0;
}}
</style>
""", unsafe_allow_html=True)


class StockAnalysisInterface:
    """Interactive interface for stock analysis"""
    
    def __init__(self):
        if 'combo' not in st.session_state:
            with st.spinner('Initializing analysis system...'):
                st.session_state.combo = AnalysisCombo(use_cache=True)
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
    
    def run(self):
        """Main interface function"""
        st.title("📊 Stock Analysis Dashboard")
        st.markdown("*Comprehensive stock analysis combining DCF valuation, technical analysis, and analyst estimates*")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selected view
        if st.session_state.get('selected_view') == 'quadrant':
            self.render_quadrant_dashboard()
        elif st.session_state.get('selected_view') == 'stock_price':
            self.render_stock_price_detail()
        elif st.session_state.get('selected_view') == 'dcf_eval':
            self.render_dcf_evaluation_detail()
        elif st.session_state.get('selected_view') == 'comp_info':
            self.render_company_info_detail()
        elif st.session_state.get('selected_view') == 'forecast':
            self.render_forecast_detail()
        elif st.session_state.get('selected_view') == 'overview':
            self.render_overview()
        elif st.session_state.get('selected_view') == 'individual':
            self.render_individual_analysis()
        elif st.session_state.get('selected_view') == 'comparison':
            self.render_comparison()
        else:
            self.render_quadrant_dashboard()
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Navigation
        view = st.sidebar.selectbox(
            "Select View",
            options=['quadrant', 'overview', 'individual', 'comparison'],
            format_func=lambda x: {
                'quadrant': '🏠 Main Dashboard (2x2)',
                'overview': '📈 Market Overview',
                'individual': '🔍 Individual Stock',
                'comparison': '⚖️ Stock Comparison'
            }[x]
        )
        st.session_state.selected_view = view
        
        # Analysis controls
        st.sidebar.subheader("⚡ Analysis")
        
        # Run analysis buttons
        if st.sidebar.button("🚀 Run Full Analysis", type="primary"):
            self.run_full_analysis()
        
        # Individual stock analysis
        if view == 'individual':
            tickers = st.session_state.combo.data_manager.get_stock_list()
            selected_ticker = st.sidebar.selectbox("Select Stock", tickers)
            st.session_state.selected_ticker = selected_ticker
            
            if st.sidebar.button(f"📊 Analyze {selected_ticker}"):
                self.run_individual_analysis(selected_ticker)
        
        # Data controls
        st.sidebar.subheader("💾 Data")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Save Results"):
                self.save_results()
        with col2:
            if st.button("Load Results"):
                self.load_results()
        
        # Status
        st.sidebar.subheader("📋 Status")
        num_analyzed = len(st.session_state.analysis_results)
        total_stocks = len(st.session_state.combo.data_manager.get_stock_list())
        st.sidebar.info(f"Analyzed: {num_analyzed}/{total_stocks} stocks")
        
        if num_analyzed > 0:
            last_update = max(result.analysis_date for result in st.session_state.analysis_results.values())
            st.sidebar.success(f"Last update: {last_update}")
    
    def run_full_analysis(self):
        """Run analysis for all stocks"""
        with st.spinner('Running comprehensive analysis for all stocks...'):
            results = st.session_state.combo.run_all_stocks()
            st.session_state.analysis_results = results
            st.success("✅ Full analysis completed!")
            st.rerun()
    
    def run_individual_analysis(self, ticker: str):
        """Run analysis for a single stock"""
        with st.spinner(f'Analyzing {ticker}...'):
            result = st.session_state.combo.run_comprehensive_analysis(ticker)
            st.session_state.analysis_results[ticker] = result
            st.success(f"✅ Analysis completed for {ticker}!")
            st.rerun()
    
    def save_results(self):
        """Save analysis results"""
        st.session_state.combo.combined_results = st.session_state.analysis_results
        st.session_state.combo.save_results()
        st.sidebar.success("Results saved!")
    
    def load_results(self):
        """Load analysis results"""
        try:
            results = st.session_state.combo.load_results()
            st.session_state.analysis_results = results
            st.sidebar.success("Results loaded!")
            st.rerun()
        except FileNotFoundError:
            st.sidebar.error("No saved results found!")
    
    def render_quadrant_dashboard(self):
        """Render the main 2x2 quadrant dashboard"""
        st.header("🏠 Main Dashboard")
        
        # Stock selector
        if not st.session_state.analysis_results:
            st.warning("No analysis data available. Run analysis first.")
            return
        
        available_tickers = list(st.session_state.analysis_results.keys())
        selected_ticker = st.selectbox("Select Stock", available_tickers, key="quadrant_stock")
        
        if not selected_ticker:
            return
        
        analysis = st.session_state.analysis_results[selected_ticker]
        
        # Create 2x2 grid layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Top-left: Stock Price
            st.subheader("📈 Stock Price")
            
            # Key price metrics
            col_price1, col_price2 = st.columns(2)
            with col_price1:
                st.metric("Current", f"${analysis.current_price:.2f}")
            with col_price2:
                change_color = "normal" if analysis.price_change_1d >= 0 else "inverse"
                st.metric("1D Change", f"{analysis.price_change_1d:+.2f}%", delta_color=change_color)
            
            # Full stock chart
            self.render_price_chart(selected_ticker)
        
        with col2:
            # Top-right: DCF / Corporate Evaluation
            st.subheader("💰 DCF / Corporate Evaluation")
            
            from analysis_combo import run_scenario_analysis, load_company_fundamentals, scenario_presets, run_multiple_scenarios
            import pandas as pd, numpy as np
            
            # --- Scenario selector ---
            selected_scenario = st.selectbox(
                "Scenario", 
                ["Base", "Bear", "Bull"], 
                index=1,  # Default to "Base"
                key="dcf_scenario_selector"
            )

            # --- Determine ticker and load real fundamentals ---
            ticker = selected_ticker or "AAPL"
            facts = load_company_fundamentals(ticker)
            
            # --- Extract fundamentals with safety ---
            shares_outstanding = to_num(facts.get("shares_outstanding")) or 0
            total_debt = to_num(facts.get("total_debt")) or 0
            cash = to_num(facts.get("cash_and_equivalents")) or 0
            revenue_base = to_num(facts.get("revenue_ttm")) or 0
            net_debt = total_debt - cash
            industry_text = facts.get("industry_text")

            # --- Get scenario presets and select the chosen one ---
            presets = scenario_presets(industry_text)
            scenario = presets[selected_scenario].copy()
            
            # --- Override with real fundamentals ---
            scenario.update({
                "revenue_base": revenue_base or scenario["revenue_base"],
                "shares": shares_outstanding or scenario["shares"],
                "net_debt": net_debt or scenario["net_debt"]
            })

            # --- Run scenario analysis ---
            ticker = str(locals().get("selected_ticker") or st.session_state.get("selected_ticker") or "TICK")
            result = run_scenario_analysis(ticker, scenario, industry_text)

            # --- Extract results ---
            proj = result["projection_df"]
            val = result["valuation"]  # dict with EV, equity_value, price_per_share
            breakdown = result.get("breakdown", {})

            # --- Build presentational DCF table ---
            df = proj.copy()
            
            # Ensure aliases
            if "delta_nwc" in df.columns and "NWC_Change" not in df.columns:
                df["NWC_Change"] = df["delta_nwc"]
            if "capex" in df.columns and "CapEx" not in df.columns:
                df["CapEx"] = df["capex"]
            if "fcf" in df.columns and "FCF" not in df.columns:
                df["FCF"] = df["fcf"]
            if "Taxes" in df.columns and "Tax" not in df.columns:
                df["Tax"] = df["Taxes"]

            # Coerce numeric columns
            for c in ["Revenue", "EBITDA", "EBIT", "Tax", "Depreciation", "CapEx", "FCF", "NWC_Change"]:
                if c in df.columns:
                    df[c] = df[c].map(lambda v: to_num(v) or 0.0)

            # Add year column for display
            df["t"] = np.arange(1, len(df) + 1, dtype=float)

            # --- Render metrics using values from DCF engine ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Enterprise Value (EV)", f"${val['EV'] / 1000:.2f}B")
            m2.metric("Equity Value", f"${val['equity_value'] / 1000:.2f}B")
            m3.metric("Price per Share", f"${val['price_per_share']:.2f}")

            # --- Scenario comparison table ---
            with st.expander("Scenario Comparison", expanded=False):
                try:
                    # Run all scenarios for comparison
                    comparison_results = run_multiple_scenarios(ticker, presets, industry_text)
                    
                    # Build comparison DataFrame
                    comparison_data = []
                    for scenario_name, result in comparison_results.items():
                        comparison_data.append({
                            "Scenario": scenario_name,
                            "WACC": f"{result['wacc']:.1%}",
                            "Terminal g": f"{result['g']:.1%}",
                            "Price/Share": f"${result['valuation']['price_per_share']:.2f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.caption(f"Comparison unavailable: {str(e)}")

            # Show warning if price seems unrealistic
            if 'warning' in breakdown:
                st.warning(breakdown['warning'])

            with st.expander("📊 DCF Projection (formatted)", expanded=True):
                # ------------- 1) Build a clean numeric projection table from the actual df -------------
                value_cols = ["Revenue","EBITDA","EBIT","Tax","Depreciation","CapEx","FCF","NWC_Change"]
                proj = df.copy()

                # Keep only the columns we care about and coerce to float safely
                keep = [c for c in value_cols if c in proj.columns]
                proj = proj[keep].applymap(lambda x: to_num(x) or 0.0)

                # Derive fiscal years: use a Year column if present; else current_year+1 .. +len
                from datetime import datetime
                start_year = datetime.now().year + 1
                fiscal_years = [str(start_year + i) for i in range(len(proj))]
                proj.insert(0, "Fiscal Year", fiscal_years)

                # ------------- 2) Decide display unit automatically (B, M, or $) -------------
                # Compute magnitude from actual numbers (avoid any constant placeholders)
                max_abs = 0.0
                if not proj.empty and len(keep) > 0:
                    max_abs = float(np.nanmax(np.abs(proj[keep].values)))

                if max_abs >= 1e9:
                    unit_divisor = 1e9
                    unit_label = "$B"
                    fmt = lambda v: f"${(v/unit_divisor):,.2f}B"
                elif max_abs >= 1e6:
                    unit_divisor = 1e6
                    unit_label = "$M"
                    fmt = lambda v: f"${(v/unit_divisor):,.2f}M"
                else:
                    unit_divisor = 1.0
                    unit_label = "$"
                    fmt = lambda v: f"${v:,.0f}"

                # ------------- 3) Build the transposed, formatted display table -------------
                # Set index to Fiscal Year then transpose → Years across columns, metrics down rows
                numeric_for_csv = proj.copy()  # keep a clean numeric copy for download
                proj_indexed = proj.set_index("Fiscal Year")

                # Create a formatted copy for display
                display_df = proj_indexed.copy()
                for c in keep:
                    display_df[c] = display_df[c].apply(fmt)

                # Transpose so rows = metrics, columns = fiscal years
                display_t = display_df[keep].T
                display_t.index.name = "Metric"

                # ------------- 4) Render table only (no chart) -------------
                st.caption(f"DCF Projection (in {unit_label.replace('$', '$B')}) — fiscal years on columns, metrics on rows")
                st.dataframe(display_t, use_container_width=True, hide_index=False)

                # ------------- 5) CSV download (unformatted numeric values + unit info) -------------
                export_csv = numeric_for_csv.rename(columns={c: f"{c} ({unit_label})" for c in keep})
                st.download_button(
                    "Download DCF Projection CSV",
                    data=export_csv.to_csv(index=False),
                    file_name="DCF_projection.csv",
                    mime="text/csv"
                )
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Bottom-left: Company Info / Estimates
            st.subheader("🏢 Company Info / Estimates")
            
            # Company info table
            company_data = {
                'Metric': [
                    'Company Name',
                    'Market Cap',
                    'P/E Ratio',
                    'Analyst Target',
                    'Analyst Upside',
                    'Number of Analysts',
                    'Analyst Rating'
                ],
                'Value': [
                    analysis.company_name,
                    f"${fdiv(analysis.market_cap, 1e9):.1f}B" if to_num(analysis.market_cap) else "N/A",
                    f"{to_num(analysis.pe_ratio):.1f}" if to_num(analysis.pe_ratio) else "N/A",
                    f"${to_num(analysis.target_median):.2f}",
                    f"{to_num(analysis.upside_to_target):+.1f}%",
                    f"{analysis.number_of_analysts}",
                    f"{analysis.analyst_rating:.1f}/5.0"
                ]
            }
            
            company_df = pd.DataFrame(company_data)
            st.dataframe(company_df, use_container_width=True, hide_index=True)
        
        with col4:
            # Bottom-right: Forecast / Recommendation
            st.subheader("🔮 Forecast / Recommendation")
            
            # Recommendation table
            recommendation_data = {
                'Analysis': [
                    'Overall Signal',
                    'Signal Strength',
                    'Risk Level',
                    'Technical Signal',
                    'Bollinger Position',
                    'DCF Analysis',
                    'Analyst Consensus',
                    'Recommendation'
                ],
                'Value': [
                    analysis.overall_signal,
                    f"{analysis.signal_strength}/10",
                    analysis.risk_level,
                    analysis.technical_signal,
                    analysis.bollinger_position,
                    f"{'Undervalued' if to_num(analysis.dcf_upside) and to_num(analysis.dcf_upside) > 0 else 'Overvalued'} by {abs(to_num(analysis.dcf_upside) or 0):.1f}%",
                    f"{'Above' if to_num(analysis.upside_to_target) and to_num(analysis.upside_to_target) > 0 else 'Below'} target by {abs(to_num(analysis.upside_to_target) or 0):.1f}%",
                    analysis.recommendation
                ]
            }
            
            recommendation_df = pd.DataFrame(recommendation_data)
            st.dataframe(recommendation_df, use_container_width=True, hide_index=True)
            
            # DCF vs Market Reality Explanation
            with st.expander("🧠 DCF vs. Market Reality — Explanation", expanded=False):
                st.markdown("""
                ### Understanding valuation gaps between DCF results and market prices

                The discounted cash flow (DCF) model provides a theoretically sound estimate of *intrinsic value*, 
                based purely on projected fundamentals and the cost of capital. However, equity markets price stocks 
                through a much broader lens — incorporating **investor sentiment**, **risk appetite**, **macro expectations**, 
                **liquidity**, and **behavioral dynamics** that the DCF framework does not capture directly.

                In practice, DCF valuations and market prices often diverge for several key reasons:

                • **Risk perception and required return:**  
                  The market may apply a higher or lower implicit discount rate than the model's WACC, depending on sentiment 
                  and recent volatility. Stocks with strong momentum or perceived safety (e.g., Apple) often trade at premiums 
                  not justified by conservative DCF assumptions.

                • **Growth optionality and intangibles:**  
                  A DCF captures only measurable cash flows, but investors also price *option value* — future innovations, 
                  ecosystem effects, or strategic assets that have uncertain payoffs. For firms like Apple or Nvidia, this 
                  optionality can be enormous and makes the market price exceed the purely modeled fair value.

                • **Macro and interest rate cycles:**  
                  When interest rates are low, market valuations inflate because investors discount future cash flows more lightly. 
                  Conversely, when rates rise, DCF-based valuations (anchored on a normalized WACC) can appear low relative 
                  to the market's still-high multiples.

                • **Behavioral and liquidity effects:**  
                  Market prices also reflect flows, investor preferences, and narrative momentum rather than fundamentals alone. 
                  "Quality" or "AI" themes, for example, can cause persistent deviations from intrinsic value across entire sectors.

                • **Model uncertainty:**  
                  Even robust DCFs rely on assumptions — growth, margins, CapEx, tax rates, and terminal *g* — that have compounding 
                  effects over long horizons. Small changes in these can swing valuations ±20–30%, and markets continuously re-price 
                  these expectations.

                **In summary:**  
                The DCF model should be interpreted not as a single-point truth but as an **anchor of fundamental value** within a broader market context. 
                Large gaps between DCF and market prices do not necessarily imply model error — they highlight where sentiment, optionality, 
                or macro narratives are driving valuations beyond purely cash-flow-based logic.
                """)
        

    def render_overview(self):
        """Render market overview"""
        st.header("📈 Market Overview")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis data available. Run analysis first.")
            return
        
        results = st.session_state.analysis_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            buy_signals = sum(1 for r in results.values() if 'BUY' in r.overall_signal)
            st.metric("📈 Buy Signals", buy_signals, f"out of {len(results)}")
        
        with col2:
            dcf_upsides = [to_num(r.dcf_upside) or 0 for r in results.values()]
            avg_dcf_upside = np.mean(dcf_upsides)
            st.metric("💰 Avg DCF Upside", f"{avg_dcf_upside:.1f}%")
        
        with col3:
            analyst_upsides = [to_num(r.upside_to_target) or 0 for r in results.values()]
            avg_analyst_upside = np.mean(analyst_upsides)
            st.metric("🎯 Avg Analyst Upside", f"{avg_analyst_upside:.1f}%")
        
        with col4:
            high_risk = sum(1 for r in results.values() if r.risk_level == 'HIGH')
            st.metric("⚠️ High Risk Stocks", high_risk, f"out of {len(results)}")
        
        # Main overview table
        st.subheader("📊 Stock Summary")
        df = self.create_overview_dataframe()
        
        # Apply styling
        def style_signal(val):
            if 'BUY' in str(val):
                return 'background-color: #d4edda; color: #155724'
            elif 'SELL' in str(val):
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        def style_percentage(val):
            try:
                num_val = float(val)
                if num_val > 0:
                    return 'color: #28a745; font-weight: bold'
                elif num_val < 0:
                    return 'color: #dc3545; font-weight: bold'
            except:
                pass
            return ''
        
        styled_df = df.style.applymap(style_signal, subset=['Signal']) \
                          .applymap(style_percentage, subset=['DCF ↑%', 'Analyst ↑%'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_signal_distribution_chart()
        
        with col2:
            self.render_risk_return_chart()
    
    def render_individual_analysis(self):
        """Render individual stock analysis"""
        if 'selected_ticker' not in st.session_state:
            st.warning("Please select a stock from the sidebar.")
            return
        
        ticker = st.session_state.selected_ticker
        
        if ticker not in st.session_state.analysis_results:
            st.warning(f"No analysis data for {ticker}. Run analysis first.")
            return
        
        analysis = st.session_state.analysis_results[ticker]
        
        # Header
        st.header(f"🔍 {analysis.company_name} ({ticker})")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${analysis.current_price}", 
                     f"{analysis.price_change_1d:+.2f}%")
        
        with col2:
            st.metric("DCF Value", f"${analysis.dcf_value}", 
                     f"{analysis.dcf_upside:+.1f}%")
        
        with col3:
            st.metric("Analyst Target", f"${analysis.target_median:.2f}", 
                     f"{analysis.upside_to_target:+.1f}%")
        
        with col4:
            signal_color = {"STRONG BUY": "🟢", "BUY": "🟡", "HOLD": "🔵", 
                          "SELL": "🟠", "STRONG SELL": "🔴"}
            st.metric("Overall Signal", 
                     f"{signal_color.get(analysis.overall_signal, '⚪')} {analysis.overall_signal}")
        
        with col5:
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            st.metric("Risk Level", 
                     f"{risk_color.get(analysis.risk_level, '⚪')} {analysis.risk_level}")
        
        # Detailed sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Price Chart", "🎯 Valuation", "🔧 Technical", "📊 Fundamentals", "💡 Summary", "🎲 Scenarios"])
        
        with tab1:
            self.render_price_chart(ticker)
        
        with tab2:
            self.render_valuation_section(analysis)
        
        with tab3:
            self.render_technical_section(analysis)
        
        with tab4:
            self.render_fundamentals_section(analysis)
        
        with tab5:
            self.render_summary_section(analysis)
        
        with tab6:
            self.render_scenario_analysis(ticker, analysis.current_price)
    
    def render_comparison(self):
        """Render stock comparison view"""
        st.header("⚖️ Stock Comparison")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis data available. Run analysis first.")
            return
        
        # Multi-select for comparison
        available_tickers = list(st.session_state.analysis_results.keys())
        selected_tickers = st.multiselect(
            "Select stocks to compare",
            available_tickers,
            default=available_tickers[:3] if len(available_tickers) >= 3 else available_tickers
        )
        
        if len(selected_tickers) < 2:
            st.warning("Please select at least 2 stocks for comparison.")
            return
        
        # Comparison metrics
        comparison_data = []
        for ticker in selected_tickers:
            analysis = st.session_state.analysis_results[ticker]
            comparison_data.append({
                'Ticker': ticker,
                'Company': analysis.company_name,
                'Price': analysis.current_price,
                'DCF Value': analysis.dcf_value,
                'DCF Upside %': analysis.dcf_upside,
                'Analyst Target': analysis.target_median,
                'Analyst Upside %': analysis.upside_to_target,
                'Signal': analysis.overall_signal,
                'Signal Strength': analysis.signal_strength,
                'Risk Level': analysis.risk_level,
                'PE Ratio': analysis.pe_ratio,
                'Beta': analysis.beta
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_valuation_comparison_chart(comparison_df)
        
        with col2:
            self.render_signal_strength_chart(comparison_df)
    
    def create_overview_dataframe(self) -> pd.DataFrame:
        """Create overview dataframe for all stocks"""
        data = []
        for analysis in st.session_state.analysis_results.values():
            data.append({
                'Ticker': analysis.ticker,
                'Company': analysis.company_name,
                'Price': f"${analysis.current_price:.2f}",
                'DCF Value': f"${analysis.dcf_value:.2f}",
                'DCF ↑%': f"{analysis.dcf_upside:+.1f}%",
                'Target': f"${analysis.target_median:.2f}",
                'Analyst ↑%': f"{analysis.upside_to_target:+.1f}%",
                'Signal': analysis.overall_signal,
                'Strength': f"{analysis.signal_strength}/10",
                'Risk': analysis.risk_level
            })
        
        return pd.DataFrame(data)
    
    def render_price_chart(self, ticker: str):
        """Render price chart with Bollinger Bands and ATR"""
        ohlcv_data = st.session_state.combo.data_manager.export_ohlcv_data(ticker)
        analysis = st.session_state.analysis_results[ticker]
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Bollinger Bands', 'ATR (Average True Range)'),
            row_heights=[0.7, 0.3]
        )
        
        # Price candlesticks
        fig.add_trace(
            go.Candlestick(
                x=ohlcv_data['Date'],
                open=ohlcv_data['Open'],
                high=ohlcv_data['High'],
                low=ohlcv_data['Low'],
                close=ohlcv_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        tech_data = st.session_state.combo.data_manager.export_technical_indicators(ticker)
        if 'BB_Upper' in tech_data.columns and 'BB_Lower' in tech_data.columns and 'BB_Middle' in tech_data.columns:
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=tech_data['Date'],
                    y=tech_data['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='rgba(255,0,0,0.3)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data['Date'],
                    y=tech_data['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='rgba(255,0,0,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data['Date'],
                    y=tech_data['BB_Middle'],
                    name='BB Middle (SMA 20)',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # ATR (Average True Range)
        if 'ATR' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data['Date'],
                    y=tech_data['ATR'],
                    name='ATR',
                    line=dict(color='orange', width=2),
                    fill='tonexty',
                    fillcolor='rgba(255,165,0,0.2)'
                ),
                row=2, col=1
            )
        
        # Add current ATR value as annotation
        if 'ATR' in tech_data.columns and not tech_data['ATR'].empty:
            current_atr = tech_data['ATR'].iloc[-1]
            current_price = ohlcv_data['Close'].iloc[-1]
            atr_percentage = (current_atr / current_price) * 100
            
            fig.add_annotation(
                x=tech_data['Date'].iloc[-1],
                y=current_atr,
                text=f"ATR: ${current_atr:.2f}<br>({atr_percentage:.1f}% of price)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="orange",
                row=2, col=1
            )
        
        fig.update_layout(
            title=f"{ticker} - Price Chart with Bollinger Bands & ATR",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="ATR ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_valuation_section(self, analysis: StockAnalysis):
        """Render valuation analysis section"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 DCF Valuation")
            
            # Valuation metrics
            valuation_data = {
                'Current Price': f"${analysis.current_price:.2f}",
                'DCF Fair Value': f"${analysis.dcf_value:.2f}",
                'Upside/Downside': f"{analysis.dcf_upside:+.1f}%",
                'WACC': f"{analysis.wacc:.2%}",
                'Terminal Growth': f"{analysis.terminal_growth:.2%}"
            }
            
            for key, value in valuation_data.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("🎯 Analyst Estimates")
            
            # Analyst data
            analyst_data = {
                'Price Target High': f"${analysis.target_high:.2f}",
                'Price Target Median': f"${analysis.target_median:.2f}",
                'Price Target Low': f"${analysis.target_low:.2f}",
                'Analyst Rating': f"{analysis.analyst_rating:.1f}/5.0",
                'Number of Analysts': f"{analysis.number_of_analysts}"
            }
            
            for key, value in analyst_data.items():
                st.metric(key, value)
        
        # Valuation comparison chart
        fig = go.Figure(go.Bar(
            x=['Current Price', 'DCF Value', 'Analyst Target'],
            y=[analysis.current_price, analysis.dcf_value, analysis.target_median],
            marker_color=['blue', 'green', 'orange'],
            text=[f"${analysis.current_price:.2f}", f"${analysis.dcf_value:.2f}", f"${analysis.target_median:.2f}"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Valuation Comparison",
            yaxis_title="Price ($)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_technical_section(self, analysis: StockAnalysis):
        """Render technical analysis section"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Technical Indicators")
            
            technical_data = {
                'RSI': f"{analysis.rsi:.1f}",
                'MACD Signal': analysis.macd_signal,
                'MA Trend': analysis.ma_trend,
                'Bollinger Position': analysis.bollinger_position,
                'Stochastic Signal': analysis.stochastic_signal
            }
            
            for key, value in technical_data.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("🎯 Risk Management")
            
            risk_data = {
                'Stop Loss': f"${analysis.stop_loss:.2f}",
                'Technical Target': f"${analysis.target_price_technical:.2f}",
                'Support Level': f"${analysis.support_level:.2f}" if analysis.support_level else "N/A",
                'Resistance Level': f"${analysis.resistance_level:.2f}" if analysis.resistance_level else "N/A"
            }
            
            for key, value in risk_data.items():
                st.metric(key, value)
        
        # Technical signal gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=analysis.technical_strength,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Technical Signal Strength"},
            delta={'reference': 5},
            gauge={'axis': {'range': [None, 10]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 3], 'color': "lightgray"},
                       {'range': [3, 7], 'color': "yellow"},
                       {'range': [7, 10], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 8}}))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_fundamentals_section(self, analysis: StockAnalysis):
        """Render fundamental analysis section"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💼 Valuation Ratios")
            
            ratios = {
                'P/E Ratio': f"{analysis.pe_ratio:.1f}" if analysis.pe_ratio else "N/A",
                'Forward P/E': f"{analysis.forward_pe:.1f}" if analysis.forward_pe else "N/A",
                'PEG Ratio': f"{analysis.peg_ratio:.2f}" if analysis.peg_ratio else "N/A",
                'P/B Ratio': f"{analysis.pb_ratio:.2f}" if analysis.pb_ratio else "N/A",
                'P/S Ratio': f"{analysis.ps_ratio:.2f}" if analysis.ps_ratio else "N/A"
            }
            
            for key, value in ratios.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📊 Financial Health")
            
            health_metrics = {
                'Market Cap': f"${analysis.market_cap/1e9:.1f}B" if analysis.market_cap else "N/A",
                'Debt/Equity': f"{analysis.debt_to_equity:.2f}" if analysis.debt_to_equity else "N/A",
                'ROE': f"{analysis.roe:.1%}" if analysis.roe else "N/A",
                'Revenue Growth': f"{analysis.revenue_growth:.1%}" if analysis.revenue_growth else "N/A",
                'Beta': f"{analysis.beta:.2f}" if analysis.beta else "N/A"
            }
            
            for key, value in health_metrics.items():
                st.metric(key, value)
    
    def render_summary_section(self, analysis: StockAnalysis):
        """Render analysis summary"""
        st.subheader("💡 Investment Summary")
        
        # Overall recommendation
        signal_colors = {
            'STRONG BUY': '#28a745',
            'BUY': '#6f42c1', 
            'HOLD': '#fd7e14',
            'SELL': '#dc3545',
            'STRONG SELL': '#721c24'
        }
        
        color = signal_colors.get(analysis.overall_signal, '#6c757d')
        
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin: 0; text-align: center;">{analysis.overall_signal}</h2>
            <h3 style="margin: 5px 0; text-align: center;">Signal Strength: {analysis.signal_strength}/10</h3>
            <p style="margin: 5px 0; text-align: center; font-size: 18px;">{analysis.recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key points
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Valuation**")
            dcf_status = "Undervalued" if analysis.dcf_upside > 0 else "Overvalued"
            analyst_status = "Above target" if analysis.upside_to_target > 0 else "Below target"
            
            st.write(f"• DCF: {dcf_status} by {abs(analysis.dcf_upside):.1f}%")
            st.write(f"• Analysts: {analyst_status} by {abs(analysis.upside_to_target):.1f}%")
        
        with col2:
            st.markdown("**📈 Technical**")
            st.write(f"• Primary trend: {analysis.ma_trend}")
            st.write(f"• RSI: {analysis.rsi:.1f} ({'Oversold' if analysis.rsi < 30 else 'Overbought' if analysis.rsi > 70 else 'Neutral'})")
            st.write(f"• Technical signal: {analysis.technical_signal}")
        
        with col3:
            st.markdown("**⚠️ Risk Factors**")
            st.write(f"• Risk level: {analysis.risk_level}")
            st.write(f"• Beta: {analysis.beta:.2f}" if analysis.beta else "• Beta: N/A")
            st.write(f"• Technical confidence: {analysis.technical_confidence:.0%}")
    
    def render_signal_distribution_chart(self):
        """Render signal distribution chart"""
        signals = [r.overall_signal for r in st.session_state.analysis_results.values()]
        signal_counts = pd.Series(signals).value_counts()
        
        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Signal Distribution"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_return_chart(self):
        """Render risk vs return scatter plot"""
        data = []
        for analysis in st.session_state.analysis_results.values():
            data.append({
                'Ticker': analysis.ticker,
                'DCF_Upside': analysis.dcf_upside,
                'Beta': analysis.beta if analysis.beta else 1.0,
                'Signal': analysis.overall_signal
            })
        
        df = pd.DataFrame(data)
        
        fig = px.scatter(
            df, x='Beta', y='DCF_Upside',
            color='Signal',
            size='DCF_Upside',
            hover_data=['Ticker'],
            title="Risk vs Return (Beta vs DCF Upside)"
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=1, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_valuation_comparison_chart(self, comparison_df: pd.DataFrame):
        """Render valuation comparison chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Price',
            x=comparison_df['Ticker'],
            y=comparison_df['Price']
        ))
        
        fig.add_trace(go.Bar(
            name='DCF Value',
            x=comparison_df['Ticker'],
            y=comparison_df['DCF Value']
        ))
        
        fig.add_trace(go.Bar(
            name='Analyst Target',
            x=comparison_df['Ticker'],
            y=comparison_df['Analyst Target']
        ))
        
        fig.update_layout(
            title="Price vs Valuation Comparison",
            barmode='group',
            yaxis_title="Price ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_strength_chart(self, comparison_df: pd.DataFrame):
        """Render signal strength comparison"""
        fig = go.Figure(go.Bar(
            x=comparison_df['Ticker'],
            y=comparison_df['Signal Strength'],
            text=comparison_df['Signal'],
            textposition='auto',
            marker_color=['green' if 'BUY' in signal else 'red' if 'SELL' in signal else 'orange' 
                         for signal in comparison_df['Signal']]
        ))
        
        fig.update_layout(
            title="Signal Strength Comparison",
            yaxis_title="Signal Strength (1-10)",
            yaxis=dict(range=[0, 10])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_scenario_analysis(self, ticker: str, current_price: float):
        """Render Bear/Base/Bull scenario analysis with sensitivity"""
        st.subheader("🎲 DCF Scenario Analysis")
        
        # Top controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            valuation_mode = st.radio("Valuation Mode", ["Perpetuity", "Exit multiple"], key=f"mode_{ticker}")
        
        with col2:
            wacc = st.number_input("WACC (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1, key=f"wacc_{ticker}") / 100
        
        with col3:
            terminal_g = st.number_input("Terminal g (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key=f"g_{ticker}") / 100
        
        # Additional inputs
        col4, col5, col6 = st.columns(3)
        
        with col4:
            exit_multiple = st.number_input("Exit Multiple", min_value=1.0, max_value=50.0, value=10.0, step=0.5, key=f"exit_{ticker}") if valuation_mode == "Exit multiple" else 10.0
        
        with col5:
            ebitda_margin_now = st.slider("EBITDA Margin Now (%)", min_value=0.0, max_value=50.0, value=25.0, step=0.5, key=f"margin_now_{ticker}") / 100
        
        with col6:
            margin_target = st.slider("Margin Target (%)", min_value=0.0, max_value=50.0, value=30.0, step=0.5, key=f"margin_target_{ticker}") / 100
        
        # More inputs
        col7, col8, col9 = st.columns(3)
        
        with col7:
            capex_pct = st.slider("CAPEX % Sales", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key=f"capex_{ticker}") / 100
        
        with col8:
            nwc_pct = st.slider("NWC % Sales", min_value=0.0, max_value=30.0, value=10.0, step=0.5, key=f"nwc_{ticker}") / 100
        
        with col9:
            tax_rate = st.slider("Tax Rate (%)", min_value=0.0, max_value=50.0, value=21.0, step=0.5, key=f"tax_{ticker}") / 100
        
        # Years input
        years = st.slider("Projection Years", min_value=3, max_value=10, value=5, key=f"years_{ticker}")
        
        # Build three independent scenario dicts
        base_scenario = {
            'years': years,
            'revenue_base': 1000000000.0,  # $1B base revenue
            'growth_path': [0.05, 0.05, 0.05, 0.05, 0.05],  # 5% flat growth
            'ebitda_margin_now': ebitda_margin_now,
            'margin_target': margin_target,
            'tax_rate': tax_rate,
            'capex_pct_sales': capex_pct,
            'nwc_pct_sales': nwc_pct,
            'depr_pct_sales': 0.03,
            'shares': 1000000000.0,  # 1B shares
            'net_debt': 500000000.0,  # $500M net debt
            'wacc': wacc,
            'terminal_g': terminal_g,
            'mode': 'perpetuity' if valuation_mode == "Perpetuity" else 'exit',
            'exit_multiple': exit_multiple
        }
        
        # Create independent copies for Bear and Bull
        bear_scenario = base_scenario.copy()
        bear_scenario.update({
            'growth_path': [0.02, 0.02, 0.02, 0.02, 0.02],  # Lower growth
            'margin_target': margin_target * 0.8,  # Margin compression
            'wacc': wacc + 0.02,  # Higher WACC
            'terminal_g': terminal_g - 0.005  # Lower terminal growth
        })
        
        bull_scenario = base_scenario.copy()
        bull_scenario.update({
            'growth_path': [0.10, 0.08, 0.06, 0.05, 0.04],  # Higher growth
            'margin_target': margin_target * 1.2,  # Margin expansion
            'wacc': wacc - 0.01,  # Lower WACC
            'terminal_g': terminal_g + 0.005  # Higher terminal growth
        })
        
        # Run scenario analysis
        if st.button("Run DCF Scenarios", key=f"run_dcf_{ticker}"):
            with st.spinner("Running DCF scenarios..."):
                try:
                    # Run each scenario
                    bear_result = st.session_state.combo.run_scenario_analysis(ticker, bear_scenario)
                    base_result = st.session_state.combo.run_scenario_analysis(ticker, base_scenario)
                    bull_result = st.session_state.combo.run_scenario_analysis(ticker, bull_scenario)
                    
                    # Store results
                    st.session_state[f'dcf_results_{ticker}'] = {
                        'Bear': bear_result,
                        'Base': base_result,
                        'Bull': bull_result
                    }
                except Exception as e:
                    st.error(f"Error running DCF scenarios: {e}")
                    return
        
        # Display results if available
        if f'dcf_results_{ticker}' in st.session_state:
            results = st.session_state[f'dcf_results_{ticker}']
            
            st.markdown("### 📊 DCF Scenario Results")
            
            # Create scenario cards
            col1, col2, col3 = st.columns(3)
            
            scenarios = ['Bear', 'Base', 'Bull']
            colors = ['#dc3545', '#ffc107', '#28a745']
            
            for i, (scenario_name, color) in enumerate(zip(scenarios, colors)):
                with [col1, col2, col3][i]:
                    result = results[scenario_name]
                    upside = ((result['price_per_share'] - current_price) / current_price) * 100
                    
                    st.markdown(f"""
                    <div style="background-color: {color}20; border: 2px solid {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <h3 style="color: {color}; margin: 0;">{scenario_name}</h3>
                        <h2 style="margin: 5px 0;">${result['price_per_share']:.2f}</h2>
                        <p style="margin: 5px 0;"><strong>Upside:</strong> {upside:+.1f}%</p>
                        <p style="margin: 5px 0;"><strong>EV:</strong> ${result['EV'] / 1000:.2f}B</p>
                        <p style="margin: 5px 0;"><strong>WACC:</strong> {result['wacc']:.1%}</p>
                        <p style="margin: 5px 0;"><strong>Terminal g:</strong> {result['g']:.1%}</p>
                        {f"<p style='margin: 5px 0;'><strong>Exit Multiple:</strong> {result['exit_multiple']:.1f}x</p>" if result['exit_multiple'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Comparison table
            st.markdown("### 📋 Scenario Comparison")
            comparison_data = []
            for scenario_name in scenarios:
                result = results[scenario_name]
                upside = ((result['price_per_share'] - current_price) / current_price) * 100
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Target PPS': f"${result['price_per_share']:.2f}",
                    'Upside %': f"{upside:+.1f}%",
                    'EV': f"${result['EV'] / 1000:.2f}B",
                    'WACC': f"{result['wacc']:.1%}",
                    'Terminal g': f"{result['g']:.1%}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Sensitivity Analysis
            st.markdown("### 🔍 Sensitivity Analysis")
            
            if st.button("Build Sensitivity Matrix", key=f"sensitivity_{ticker}"):
                with st.spinner("Building sensitivity matrix..."):
                    try:
                        base_result = results['Base']
                        sensitivity_df = st.session_state.combo.run_sensitivity(
                            base_result['projection_df'], 
                            base_result['wacc'], 
                            base_result['g']
                        )
                        st.session_state[f'sensitivity_{ticker}'] = sensitivity_df
                    except Exception as e:
                        st.error(f"Error building sensitivity: {e}")
            
            # Display sensitivity matrix if available
            if f'sensitivity_{ticker}' in st.session_state:
                sensitivity_df = st.session_state[f'sensitivity_{ticker}']
                
                st.markdown("**Price per Share Sensitivity (WACC vs Terminal Growth)**")
                st.dataframe(sensitivity_df.style.format("${:.2f}"), use_container_width=True)
                
                # Download button
                csv = sensitivity_df.to_csv()
                st.download_button(
                    label="Download Sensitivity Matrix as CSV",
                    data=csv,
                    file_name=f"{ticker}_sensitivity_matrix.csv",
                    mime="text/csv"
                )
                
                st.caption(f"📍 Base case: WACC={base_result['wacc']:.2%}, g={base_result['g']:.2%} | "
                          f"Grid: ±200bps WACC, ±100bps growth")
        else:
            st.info("Click 'Run DCF Scenarios' to generate Bear/Base/Bull DCF projections")
    
    
    def render_stock_price_detail(self):
        """Render detailed stock price analysis"""
        st.header("📈 Stock Price Analysis")
        
        if 'quadrant_stock' in st.session_state:
            selected_ticker = st.session_state.quadrant_stock
        else:
            available_tickers = list(st.session_state.analysis_results.keys())
            selected_ticker = st.selectbox("Select Stock", available_tickers, key="detail_stock_price")
        
        if not selected_ticker or selected_ticker not in st.session_state.analysis_results:
            st.warning("No stock selected or analysis data not available.")
            return
        
        analysis = st.session_state.analysis_results[selected_ticker]
        
        # Back button
        if st.button("← Back to Main Dashboard"):
            st.session_state.selected_view = 'quadrant'
            st.rerun()
        
        # Price chart with technical indicators
        self.render_price_chart(selected_ticker)
        
        # Price metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${analysis.current_price:.2f}")
        with col2:
            st.metric("1 Day Change", f"{analysis.price_change_1d:+.2f}%")
        with col3:
            st.metric("1 Week Change", f"{analysis.price_change_1w:+.2f}%")
        with col4:
            st.metric("1 Month Change", f"{analysis.price_change_1m:+.2f}%")
        
        # Technical levels
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("📊 Support & Resistance")
            if analysis.support_level:
                st.metric("Support Level", f"${analysis.support_level:.2f}")
            if analysis.resistance_level:
                st.metric("Resistance Level", f"${analysis.resistance_level:.2f}")
        
        with col6:
            st.subheader("🎯 Risk Management")
            st.metric("Stop Loss", f"${analysis.stop_loss:.2f}")
            st.metric("Technical Target", f"${analysis.target_price_technical:.2f}")
    
    def render_dcf_evaluation_detail(self):
        """Render detailed DCF evaluation"""
        st.header("💰 DCF / Corporate Evaluation")
        
        if 'quadrant_stock' in st.session_state:
            selected_ticker = st.session_state.quadrant_stock
        else:
            available_tickers = list(st.session_state.analysis_results.keys())
            selected_ticker = st.selectbox("Select Stock", available_tickers, key="detail_dcf_eval")
        
        if not selected_ticker or selected_ticker not in st.session_state.analysis_results:
            st.warning("No stock selected or analysis data not available.")
            return
        
        analysis = st.session_state.analysis_results[selected_ticker]
        
        # Back button
        if st.button("← Back to Main Dashboard"):
            st.session_state.selected_view = 'quadrant'
            st.rerun()
        
        # Variable resolver block - attempt to derive values from existing variables
        current_price = analysis.current_price
        shares_outstanding = None
        revenue_base = None
        net_debt = 0.0
        
        # Try to find shares_outstanding from existing variables
        if hasattr(analysis, 'market_cap') and analysis.market_cap and current_price:
            shares_outstanding = analysis.market_cap / current_price
        
        # Try to find revenue_base from existing variables
        if hasattr(analysis, 'market_cap') and analysis.market_cap and current_price:
            # Estimate revenue from market cap (rough approximation)
            revenue_base = analysis.market_cap / current_price * 0.1  # Assume 10% revenue/market cap ratio
        
        # If any required inputs are missing, render input expander
        if not current_price or not shares_outstanding or not revenue_base:
            with st.expander("Provide DCF inputs (only if missing)", expanded=True):
                col1, col2 = st.columns(2)
                current_price = current_price or col1.number_input("Current price ($)", min_value=0.0, value=0.0, step=0.01, key="dcf_curr_price")
                shares_outstanding = shares_outstanding or col2.number_input("Shares outstanding (units)", min_value=0.0, value=0.0, step=1_000_000.0, key="dcf_shares")
                revenue_base = revenue_base or st.number_input("Base revenue (last FY/TTM, in USD)", min_value=0.0, value=0.0, step=1_000_000.0, key="dcf_rev")
                
                st.caption("Tip: if your revenue is in millions, enter the full USD amount (e.g., 1.2e9 for $1.2B).")
                
                # Unit helpers
                col3, col4 = st.columns(2)
                with col3:
                    rev_units = st.selectbox("Revenue units", ["USD", "USD millions"], index=0, key="dcf_rev_units")
                with col4:
                    sh_units = st.selectbox("Shares units", ["units", "millions"], index=0, key="dcf_sh_units")
                
                # Normalize units
                if rev_units == "USD millions":
                    revenue_base *= 1e6
                if sh_units == "millions":
                    shares_outstanding *= 1e6
        
        # Final validation
        if not current_price or not shares_outstanding or not revenue_base or current_price <= 0 or shares_outstanding <= 0 or revenue_base <= 0:
            st.warning("DCF needs valid ticker, current price, shares outstanding, and base revenue. Please provide all required inputs above.")
            return
        
        # Industry defaults + knobs
        industry_key = list(calib.INDUSTRY_DEFAULTS.keys())[0]  # First available industry
        defaults = calib.INDUSTRY_DEFAULTS[industry_key]
        
        # WACC/g/mode controls
        mode = st.radio("Valuation mode", ["Perpetuity", "Exit multiple"], horizontal=True, key="dcf_mode")
        wacc = st.number_input("WACC (%)", value=10.0, min_value=4.0, max_value=20.0, step=0.25, key="dcf_wacc") / 100.0
        g = st.number_input("Terminal g (%)", value=float(defaults.get("lt_growth_cap", 0.025) * 100), min_value=0.0, max_value=6.0, step=0.25, key="dcf_g") / 100.0
        
        if mode == "Exit multiple":
            exit_multiple = st.number_input("Exit multiple (EV/EBITDA)", value=float(defaults.get("exit_multiple_hint", 10.0)), min_value=1.0, max_value=40.0, step=0.5, key="dcf_exit_mult")
        else:
            exit_multiple = None
        
        if g >= wacc:
            st.warning("Terminal growth must be < WACC")
            return
        
        # Build base inputs dict for project_operating_model
        base_inputs = {
            "years": 5,
            "revenue_base": revenue_base,
            "growth_path": [to_num(defaults.get("lt_growth_cap", 0.03)) or 0.03] * 5,
            "ebitda_margin_now": fsub(to_num(defaults.get("margin_target", 0.25)) or 0.25, 0.02),
            "margin_target": to_num(defaults.get("margin_target", 0.25)) or 0.25,
            "tax_rate": to_num(defaults.get("tax_rate", 0.22)) or 0.22,
            "capex_pct_sales": to_num(defaults.get("capex_pct_sales", 0.05)) or 0.05,
            "nwc_pct_sales": to_num(defaults.get("nwc_pct_sales", 0.03)) or 0.03,
            "depr_pct_sales": to_num(defaults.get("depr_pct_sales", 0.03)) or 0.03,
            "shares": shares_outstanding,
            "net_debt": net_debt,
        }
        
        # Build three independent scenario dicts
        bear = copy.deepcopy(base_inputs)
        base = copy.deepcopy(base_inputs)
        bull = copy.deepcopy(base_inputs)
        
        # Adjust each deterministically
        bear["margin_target"] = max(0.0, fsub(base_inputs["margin_target"], 0.015))  # -150 bps
        bear["growth_path"] = [max(0.0, fsub(x, 0.01)) for x in base_inputs["growth_path"]]  # -100 bps
        
        bull["margin_target"] = fsub(base_inputs["margin_target"], -0.015)  # +150 bps
        bull["growth_path"] = [fsub(x, -0.01) for x in base_inputs["growth_path"]]  # +100 bps
        
        # Add mode and exit_multiple
        scenario_mode = "perpetuity" if mode == "Perpetuity" else "exit"
        for dict_input in [bear, base, bull]:
            dict_input["mode"] = scenario_mode
            dict_input["wacc"] = wacc
            dict_input["terminal_g"] = g
            if scenario_mode == "exit":
                dict_input["exit_multiple"] = float(exit_multiple)
        
        # Call helpers
        if st.button("Run DCF Scenarios", key="run_dcf_detail"):
            with st.spinner("Running DCF scenarios..."):
                try:
                    res_bear = st.session_state.combo.run_scenario_analysis(selected_ticker, bear)
                    res_base = st.session_state.combo.run_scenario_analysis(selected_ticker, base)
                    res_bull = st.session_state.combo.run_scenario_analysis(selected_ticker, bull)
                    
                    # Store results
                    st.session_state[f'dcf_detail_results_{selected_ticker}'] = {
                        'Bear': res_bear,
                        'Base': res_base,
                        'Bull': res_bull
                    }
                except Exception as e:
                    st.error(f"Error running DCF scenarios: {e}")
                    return
        
        # Display results if available
        if f'dcf_detail_results_{selected_ticker}' in st.session_state:
            results = st.session_state[f'dcf_detail_results_{selected_ticker}']
            
            # Normalize results
            res_bear_norm = _normalize_dcf_result(results['Bear'])
            res_base_norm = _normalize_dcf_result(results['Base'])
            res_bull_norm = _normalize_dcf_result(results['Bull'])
            
            # Build results DataFrame
            df = pd.DataFrame([
                {
                    "Scenario": "Bear",
                    "Target PPS": (res_bear_norm.get('valuation') or {}).get('price_per_share', 0),
                    "Upside %": ((res_bear_norm.get('valuation') or {}).get('price_per_share', 0) / current_price - 1) * 100 if current_price and current_price > 0 else 0,
                    "WACC": wacc,
                    "g": g,
                    "Exit Multiple": exit_multiple if scenario_mode == "exit" else None
                },
                {
                    "Scenario": "Base",
                    "Target PPS": (res_base_norm.get('valuation') or {}).get('price_per_share', 0),
                    "Upside %": ((res_base_norm.get('valuation') or {}).get('price_per_share', 0) / current_price - 1) * 100 if current_price and current_price > 0 else 0,
                    "WACC": wacc,
                    "g": g,
                    "Exit Multiple": exit_multiple if scenario_mode == "exit" else None
                },
                {
                    "Scenario": "Bull",
                    "Target PPS": (res_bull_norm.get('valuation') or {}).get('price_per_share', 0),
                    "Upside %": ((res_bull_norm.get('valuation') or {}).get('price_per_share', 0) / current_price - 1) * 100 if current_price and current_price > 0 else 0,
                    "WACC": wacc,
                    "g": g,
                    "Exit Multiple": exit_multiple if scenario_mode == "exit" else None
                }
            ])
            
            # Format and display
            df_display = df.copy()
            df_display["Target PPS"] = df_display["Target PPS"].apply(lambda x: f"${x:.2f}")
            df_display["Upside %"] = df_display["Upside %"].apply(lambda x: f"{x:+.1f}%")
            df_display["WACC"] = df_display["WACC"].apply(lambda x: f"{x:.1%}")
            df_display["g"] = df_display["g"].apply(lambda x: f"{x:.1%}")
            if scenario_mode == "exit":
                df_display["Exit Multiple"] = df_display["Exit Multiple"].apply(lambda x: f"{x:.1f}x")
            else:
                df_display = df_display.drop("Exit Multiple", axis=1)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # FULL DCF TABLE (new, under the comparison table)
            st.subheader("📊 DCF Projection (in $B)")
            
            with st.expander("Full DCF table (projection + discounting)", expanded=False):
                try:
                    # Get base projection DataFrame
                    projection_df = res_base_norm["projection_df"]
                    
                    # Create a copy for the full DCF table
                    full_dcf_df = projection_df.copy()
                    
                    # Add discount factor column
                    full_dcf_df['DiscountFactor'] = 1 / (1 + wacc) ** (full_dcf_df.index + 1)
                    
                    # Add PV_FCF column
                    full_dcf_df['PV_FCF'] = full_dcf_df['FCF'] * full_dcf_df['DiscountFactor']
                    
                    # Calculate terminal value
                    if mode == "Perpetuity":
                        # Gordon Growth Model: TV = FCF_N * (1 + g) / (wacc - g)
                        final_fcf = full_dcf_df['FCF'].iloc[-1]
                        terminal_value = final_fcf * (1 + g) / (wacc - g)
                    else:  # Exit multiple
                        # TV = exit_multiple * EBITDA_N
                        final_ebitda = full_dcf_df['EBITDA'].iloc[-1]
                        terminal_value = exit_multiple * final_ebitda
                    
                    # Add terminal value column (only on last row)
                    full_dcf_df['TerminalValue'] = 0.0
                    full_dcf_df.loc[full_dcf_df.index[-1], 'TerminalValue'] = terminal_value
                    
                    # Add PV_Terminal column
                    full_dcf_df['PV_Terminal'] = full_dcf_df['TerminalValue'] * full_dcf_df['DiscountFactor']
                    
                    # Calculate summary metrics
                    total_pv_fcf = full_dcf_df['PV_FCF'].sum()
                    pv_terminal = full_dcf_df['PV_Terminal'].sum()
                    enterprise_value = total_pv_fcf + pv_terminal
                    equity_value = enterprise_value - net_debt
                    price_per_share = equity_value / shares_outstanding
                    
                    # Transpose the table and add year labels
                    # Create a transposed version with years as columns
                    transposed_df = full_dcf_df.T
                    
                    # Add fiscal year labels as column headers (2025-2030)
                    current_year = 2025
                    year_labels = [f"{current_year + i}" for i in range(len(full_dcf_df))]
                    transposed_df.columns = year_labels
                    
                    # Add a row for the year index
                    year_index_row = pd.DataFrame([list(range(current_year, current_year + len(full_dcf_df)))], 
                                                index=['Fiscal Year'], 
                                                columns=year_labels)
                    
                    # Combine year index with transposed data
                    final_df = pd.concat([year_index_row, transposed_df])
                    
                    # Helper function to format values in billions
                    def format_billions(value):
                        if pd.isna(value) or value == 0:
                            return "0.00B"
                        return f"${value / 1000:.2f}B"
                    
                    # Convert values to billions for display
                    display_df = final_df.copy()
                    # Skip the first row (Fiscal Year) and convert financial metrics to billions
                    for col in display_df.columns:
                        for idx in display_df.index[1:]:  # Skip the Fiscal Year row
                            if pd.notna(display_df.loc[idx, col]) and display_df.loc[idx, col] != 0:
                                display_df.loc[idx, col] = format_billions(display_df.loc[idx, col])
                            else:
                                display_df.loc[idx, col] = "0.00B"
                    
                    # Display the transposed DCF table with billions formatting
                    st.dataframe(display_df, use_container_width=True)
                    st.caption("DCF Projection (in $B) - Fiscal Years as columns")
                    
                    # Summary metrics strip
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Enterprise Value", f"${enterprise_value / 1000:.2f}B")
                    with col2:
                        st.metric("Equity Value", f"${equity_value / 1000:.2f}B")
                    with col3:
                        st.metric("Price per Share", f"${price_per_share:.2f}")
                    with col4:
                        st.metric("Terminal Value", f"${terminal_value / 1000:.2f}B")
                    
                    # Download button for full DCF table
                    csv_data = full_dcf_df.to_csv(index=True)
                    st.download_button(
                        "Download Full DCF Table as CSV",
                        data=csv_data,
                        file_name=f"{selected_ticker}_full_dcf_table.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error building full DCF table: {e}")
            
            # Sensitivity button
            if st.button("Build sensitivity (Base)", key="build_sensitivity_detail"):
                with st.spinner("Building sensitivity matrix..."):
                    try:
                        mat = st.session_state.combo.run_sensitivity(res_base_norm["projection_df"], wacc, g)
                        st.dataframe(mat.style.format("{:.2f}"))
                        st.download_button("Download sensitivity CSV", data=mat.to_csv(index=True), file_name=f"{selected_ticker}_sensitivity.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Error building sensitivity: {e}")
        else:
            st.info("Click 'Run DCF Scenarios' to generate Bear/Base/Bull DCF projections")
    
    def render_company_info_detail(self):
        """Render detailed company info and estimates"""
        st.header("🏢 Company Info / Estimates")
        
        if 'quadrant_stock' in st.session_state:
            selected_ticker = st.session_state.quadrant_stock
        else:
            available_tickers = list(st.session_state.analysis_results.keys())
            selected_ticker = st.selectbox("Select Stock", available_tickers, key="detail_comp_info")
        
        if not selected_ticker or selected_ticker not in st.session_state.analysis_results:
            st.warning("No stock selected or analysis data not available.")
            return
        
        analysis = st.session_state.analysis_results[selected_ticker]
        
        # Back button
        if st.button("← Back to Main Dashboard"):
            st.session_state.selected_view = 'quadrant'
            st.rerun()
        
        # Company information
        st.subheader(f"📋 {analysis.company_name} ({analysis.ticker})")
        
        # Analyst estimates section
        self.render_valuation_section(analysis)
        
        # Fundamental data
        st.subheader("📊 Fundamental Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Market Cap", f"${analysis.market_cap/1e9:.1f}B" if analysis.market_cap else "N/A")
            st.metric("P/E Ratio", f"{analysis.pe_ratio:.1f}" if analysis.pe_ratio else "N/A")
            st.metric("Forward P/E", f"{analysis.forward_pe:.1f}" if analysis.forward_pe else "N/A")
            st.metric("Beta", f"{analysis.beta:.2f}" if analysis.beta else "N/A")
        
        with col2:
            st.metric("ROE", f"{analysis.roe:.1%}" if analysis.roe else "N/A")
            st.metric("Revenue Growth", f"{analysis.revenue_growth:.1%}" if analysis.revenue_growth else "N/A")
            st.metric("Debt/Equity", f"{analysis.debt_to_equity:.2f}" if analysis.debt_to_equity else "N/A")
            st.metric("Dividend Yield", f"{analysis.dividend_yield:.1%}" if analysis.dividend_yield else "N/A")
        
        # Analyst estimates table
        st.subheader("🎯 Analyst Estimates")
        
        estimates_data = {
            'Target High': f"${analysis.target_high:.2f}",
            'Target Median': f"${analysis.target_median:.2f}",
            'Target Low': f"${analysis.target_low:.2f}",
            'Current Price': f"${analysis.current_price:.2f}",
            'Analyst Rating': f"{analysis.analyst_rating:.1f}/5.0",
            'Number of Analysts': f"{analysis.number_of_analysts}"
        }
        
        for key, value in estimates_data.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)
    
    def render_forecast_detail(self):
        """Render detailed forecast and recommendation"""
        st.header("🔮 Forecast / Recommendation")
        
        if 'quadrant_stock' in st.session_state:
            selected_ticker = st.session_state.quadrant_stock
        else:
            available_tickers = list(st.session_state.analysis_results.keys())
            selected_ticker = st.selectbox("Select Stock", available_tickers, key="detail_forecast")
        
        if not selected_ticker or selected_ticker not in st.session_state.analysis_results:
            st.warning("No stock selected or analysis data not available.")
            return
        
        analysis = st.session_state.analysis_results[selected_ticker]
        
        # Back button
        if st.button("← Back to Main Dashboard"):
            st.session_state.selected_view = 'quadrant'
            st.rerun()
        
        # Investment summary
        self.render_summary_section(analysis)
        
        # Technical analysis details
        st.subheader("📈 Technical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("RSI", f"{analysis.rsi:.1f}")
            st.metric("MACD Signal", analysis.macd_signal)
            st.metric("Moving Average Trend", analysis.ma_trend)
        
        with col2:
            st.metric("Bollinger Position", analysis.bollinger_position)
            st.metric("Stochastic Signal", analysis.stochastic_signal)
            st.metric("Technical Confidence", f"{analysis.technical_confidence:.0%}")
        
        # Risk assessment
        st.subheader("⚠️ Risk Assessment")
        
        risk_colors = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴'
        }
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Overall Risk Level", f"{risk_colors.get(analysis.risk_level, '⚪')} {analysis.risk_level}")
            st.metric("Signal Strength", f"{analysis.signal_strength}/10")
        
        with col4:
            st.metric("Technical Signal", analysis.technical_signal)
            st.metric("Overall Signal", analysis.overall_signal)
        
        # Recommendation details
        st.subheader("💡 Investment Recommendation")
        
        st.info(f"**Recommendation:** {analysis.recommendation}")
        
        # Key factors
        st.write("**Key Factors:**")
        st.write(f"• DCF Analysis: {'Undervalued' if analysis.dcf_upside > 0 else 'Overvalued'} by {abs(analysis.dcf_upside):.1f}%")
        st.write(f"• Analyst Consensus: {'Above' if analysis.upside_to_target > 0 else 'Below'} target by {abs(analysis.upside_to_target):.1f}%")
        st.write(f"• Technical Trend: {analysis.ma_trend} with {analysis.technical_signal} signal")
        st.write(f"• Risk Level: {analysis.risk_level} risk investment")


def main():
    """Main function to run the Streamlit interface"""
    interface = StockAnalysisInterface()
    interface.run()


if __name__ == "__main__":
    main()

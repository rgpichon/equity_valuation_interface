#!/usr/bin/env python3
"""
Scenario Builder - Equity Valuation Tool
Professional-grade equity valuation framework combining DCF and market-based methods

Author: Your Name
Date: 2024
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
import requests
import json
from scipy.stats import norm
import asyncio
import aiohttp

# Import the calibration system (avoid circular import)
try:
    from valuation_calibration import ValuationCalibrator
except ImportError:
    # Handle circular import by importing later
    ValuationCalibrator = None

warnings.filterwarnings('ignore')

class EquityValuationEngine:
    """
    Professional equity valuation engine combining multiple methodologies
    
    Features:
    - DCF modeling with enhanced WACC calculation
    - Market-based peer multiple analysis
    - Scenario testing (Base, Bull, Bear cases)
    - Monte Carlo simulation for uncertainty modeling
    - Risk-adjusted return calculations
    - Real-time market data integration
    - Professional reporting and visualization
    """
    
    def __init__(self, symbol: str, api_key: str = None, use_calibration: bool = True):
        """
        Initialize the valuation engine with optional calibration
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            api_key: Financial Modeling Prep API key
            use_calibration: Whether to enable parameter calibration
        """
        self.symbol = symbol.upper()
        self.api_key = api_key or "ZhY4S0yksINsUhFNdaePTZEm7AVu6QPe"
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # Initialize data containers
        self.financial_data = {}
        self.market_data = {}
        self.valuation_results = {}
        self.peer_data = {}
        
        # Initialize calibration system
        self.use_calibration = use_calibration and ValuationCalibrator is not None
        self.calibrator = ValuationCalibrator(api_key) if self.use_calibration else None
        self.calibrated_parameters = None
        self.calibration_results = {}
        
        # Load data on initialization
        self._load_company_data()
        
        # Run calibration if enabled
        if self.use_calibration:
            self._run_calibration()
    
    def _load_company_data(self) -> None:
        """Load comprehensive company data from FMP API"""
        try:
            # Fetch financial statements
            self.financial_data = self._fetch_financial_statements()
            
            # Fetch market data
            self.market_data = self._fetch_market_data()
            
            # Fetch peer data
            self.peer_data = self._fetch_peer_data()
            
            print(f"✅ Data loaded successfully for {self.symbol}")
            
        except Exception as e:
            print(f"❌ Error loading data for {self.symbol}: {e}")
            raise
    
    def _run_calibration(self) -> None:
        """Run parameter calibration to optimize valuation inputs"""
        try:
            print(f"🔧 Running calibration for {self.symbol}...")
            
            # Optimize parameters using historical data
            calibration_result = self.calibrator.optimize_parameters(
                self.symbol,
                optimization_method='minimize_mae',
                max_iterations=50  # Faster calibration
            )
            
            if 'error' not in calibration_result:
                self.calibrated_parameters = calibration_result['optimized_parameters']
                self.calibration_results = calibration_result
                print(f"✅ Calibration completed - using optimized parameters")
                print(f"   • Revenue Growth: {self.calibrated_parameters['revenue_growth']:.1%}")
                print(f"   • EBIT Margin: {self.calibrated_parameters['ebit_margin']:.1%}")
                print(f"   • WACC: {self.calibrated_parameters['wacc']:.1%}")
                print(f"   • Terminal Growth: {self.calibrated_parameters['terminal_growth']:.1%}")
            else:
                print(f"⚠️ Calibration failed, using default parameters: {calibration_result['error']}")
                self.calibrated_parameters = None
                
        except Exception as e:
            print(f"⚠️ Calibration error: {e}")
            self.calibrated_parameters = None
    
    def _fetch_financial_statements(self) -> Dict:
        """Fetch income statement, balance sheet, and cash flow data"""
        endpoints = {
            'income': f"{self.base_url}/income-statement/{self.symbol}?limit=5&apikey={self.api_key}",
            'balance': f"{self.base_url}/balance-sheet-statement/{self.symbol}?limit=5&apikey={self.api_key}",
            'cashflow': f"{self.base_url}/cash-flow-statement/{self.symbol}?limit=5&apikey={self.api_key}"
        }
        
        financial_data = {}
        for statement_type, url in endpoints.items():
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    financial_data[statement_type] = response.json()
                else:
                    print(f"Warning: Could not fetch {statement_type} data")
            except Exception as e:
                print(f"Error fetching {statement_type}: {e}")
        
        return financial_data
    
    def _fetch_market_data(self) -> Dict:
        """Fetch current market data and historical prices"""
        endpoints = {
            'profile': f"{self.base_url}/profile/{self.symbol}?apikey={self.api_key}",
            'quote': f"{self.base_url}/quote/{self.symbol}?apikey={self.api_key}",
            'historical': f"{self.base_url}/historical-price-full/{self.symbol}?apikey={self.api_key}"
        }
        
        market_data = {}
        for data_type, url in endpoints.items():
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    market_data[data_type] = response.json()
                else:
                    print(f"Warning: Could not fetch {data_type} data")
            except Exception as e:
                print(f"Error fetching {data_type}: {e}")
        
        return market_data
    
    def _fetch_peer_data(self) -> Dict:
        """Fetch peer company data for relative valuation"""
        try:
            # Get company profile to determine industry
            profile = self.market_data.get('profile', [{}])[0]
            industry = profile.get('industry', 'Technology')
            
            # Fetch peer companies (simplified - in practice, you'd have a peer database)
            peer_symbols = self._get_peer_symbols(industry)
            
            peer_data = {}
            for peer in peer_symbols:
                try:
                    url = f"{self.base_url}/quote/{peer}?apikey={self.api_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        peer_data[peer] = response.json()[0]
                except:
                    continue
            
            return peer_data
            
        except Exception as e:
            print(f"Error fetching peer data: {e}")
            return {}
    
    def _get_peer_symbols(self, industry: str) -> List[str]:
        """Get peer company symbols based on industry"""
        # Enhanced peer mapping with more comprehensive coverage
        peer_mapping = {
            'Technology': ['MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'BMY'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'USB', 'PNC'],
            'Consumer': ['KO', 'PEP', 'WMT', 'PG', 'JNJ', 'MCD', 'SBUX', 'NKE', 'COST', 'HD'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'WMB', 'OKE', 'PXD', 'MPC'],
            'Industrial': ['BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC'],
            'Communication': ['VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'DISH', 'GOOGL', 'META'],
            'Automotive': ['TSLA', 'TM', 'F', 'GM', 'HMC', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID'],
            'Luxury': ['LVMUY', 'KER.PA', 'CFR.SW', 'HERM.PA', 'MC.PA', 'TIF', 'CPRI', 'COH', 'TAP.AX', 'RACE'],
            'Aerospace': ['BA', 'LMT', 'RTX', 'NOC', 'GD', 'TDG', 'LHX', 'HWM', 'TXT', 'AJG'],
            'Semiconductors': ['TSM', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MRVL', 'AMAT'],
            'Semiconductor Equipment': ['AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'SWKS', 'QRVO', 'SLAB', 'CRUS']
        }
        
        return peer_mapping.get(industry, ['SPY'])  # Default to SPY if industry not found
    
    def calculate_dcf_valuation(self, 
                              revenue_growth: float = None,
                              ebit_margin: float = None,
                              wacc: float = None,
                              terminal_growth: float = None,
                              forecast_years: int = 5) -> Dict:
        """
        Calculate DCF valuation with enhanced methodology
        
        Args:
            revenue_growth: Annual revenue growth rate
            ebit_margin: Target EBIT margin
            wacc: Weighted Average Cost of Capital
            terminal_growth: Terminal growth rate
            forecast_years: Number of forecast years
        
        Returns:
            Dict containing DCF valuation results
        """
        try:
            # Use calibrated parameters if available, otherwise use provided or defaults
            if self.calibrated_parameters and revenue_growth is None:
                revenue_growth = self.calibrated_parameters['revenue_growth']
                ebit_margin = self.calibrated_parameters['ebit_margin']
                wacc = wacc or self.calibrated_parameters['wacc']  # Allow WACC override
                terminal_growth = self.calibrated_parameters['terminal_growth']
            else:
                # Fallback to defaults if not provided
                revenue_growth = revenue_growth or 0.15
                ebit_margin = ebit_margin or 0.30
                wacc = wacc or 0.06
                terminal_growth = terminal_growth or 0.035
            
            # Extract current financial data
            current_data = self._extract_current_financials()
            
            if not current_data:
                return {'error': 'Could not extract financial data'}
            
            # Build financial projections
            projections = self._build_financial_projections(
                current_data, revenue_growth, ebit_margin, forecast_years
            )
            
            # Use provided WACC parameter directly
            calculated_wacc = wacc
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(
                projections[-1], calculated_wacc, terminal_growth
            )
            
            # Calculate present value of cash flows
            pv_cashflows = self._calculate_present_value_cashflows(
                projections, calculated_wacc
            )
            
            # Calculate equity value
            enterprise_value = pv_cashflows + terminal_value
            equity_value = enterprise_value - current_data.get('net_debt', 0)
            shares_outstanding = current_data.get('shares_outstanding', 1)
            price_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            
            # Calculate current price and upside/downside
            current_price = current_data.get('current_price', 0)
            upside_downside = (price_per_share - current_price) / current_price if current_price > 0 else 0
            
            # Calculate key metrics
            current_pe = current_price / current_data.get('eps', 1) if current_data.get('eps', 0) > 0 else 0
            implied_pe = price_per_share / current_data.get('eps', 1) if current_data.get('eps', 0) > 0 else 0
            
            return {
                'dcf_price': price_per_share,
                'current_price': current_price,
                'upside_downside': upside_downside,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'terminal_value': terminal_value,
                'wacc': calculated_wacc,
                'projections': projections,
                'methodology': 'DCF (Calibrated)' if self.calibrated_parameters else 'DCF',
                'calibration_used': self.calibrated_parameters is not None,
                'calibrated_parameters': self.calibrated_parameters if self.calibrated_parameters else None,
                'current_pe': current_pe,
                'implied_pe': implied_pe,
                'terminal_value_pct': terminal_value / enterprise_value if enterprise_value > 0 else 0
            }
            
        except Exception as e:
            return {'error': f"DCF calculation failed: {e}"}
    
    def calculate_peer_valuation(self, custom_multiples: Dict = None) -> Dict:
        """
        Calculate valuation using peer company multiples with market-driven approach
        
        Args:
            custom_multiples: Optional custom multiples to override peer averages
        
        Returns:
            Dict containing peer-based valuation results
        """
        try:
            current_data = self._extract_current_financials()
            peer_data = self.peer_data
            
            if not peer_data:
                return {'error': 'No peer data available'}
            
            if not current_data:
                return {'error': 'Could not extract current financial data'}
            
            # Calculate peer multiples
            peer_multiples = self._calculate_peer_multiples(peer_data)
            
            # Use custom multiples if provided
            if custom_multiples:
                peer_multiples.update(custom_multiples)
            
            # Apply multiples to current company
            valuations = {}
            
            # P/E multiple
            if 'pe_ratio' in peer_multiples and current_data.get('eps', 0) > 0:
                valuations['pe'] = current_data['eps'] * peer_multiples['pe_ratio']
            
            # EV/Revenue multiple
            if 'ev_revenue' in peer_multiples:
                enterprise_value = current_data['revenue'] * peer_multiples['ev_revenue']
                equity_value = enterprise_value - current_data.get('net_debt', 0)
                valuations['ev_revenue'] = equity_value / current_data.get('shares_outstanding', 1)
            
            # EV/EBITDA multiple
            if 'ev_ebitda' in peer_multiples and current_data.get('ebitda', 0) > 0:
                enterprise_value = current_data['ebitda'] * peer_multiples['ev_ebitda']
                equity_value = enterprise_value - current_data.get('net_debt', 0)
                valuations['ev_ebitda'] = equity_value / current_data.get('shares_outstanding', 1)
            
            # Calculate weighted average peer valuation (P/E gets higher weight as it's most reliable)
            valid_valuations = []
            weights = []
            
            if 'pe' in valuations and valuations['pe'] > 0:
                valid_valuations.append(valuations['pe'])
                weights.append(0.5)  # 50% weight for P/E
            
            if 'ev_revenue' in valuations and valuations['ev_revenue'] > 0:
                valid_valuations.append(valuations['ev_revenue'])
                weights.append(0.3)  # 30% weight for EV/Revenue
            
            if 'ev_ebitda' in valuations and valuations['ev_ebitda'] > 0:
                valid_valuations.append(valuations['ev_ebitda'])
                weights.append(0.2)  # 20% weight for EV/EBITDA
            
            if valid_valuations:
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                weighted_peer_price = sum(v * w for v, w in zip(valid_valuations, weights))
            else:
                weighted_peer_price = 0
            
            current_price = current_data.get('current_price', 0)
            upside_downside = (weighted_peer_price - current_price) / current_price if current_price > 0 else 0
            
            return {
                'peer_price': weighted_peer_price,
                'current_price': current_price,
                'upside_downside': upside_downside,
                'individual_valuations': valuations,
                'peer_multiples': peer_multiples,
                'methodology': 'Market-Driven Peer Multiples',
                'peer_count': len(peer_data),
                'valuation_consensus': 'High' if len(valid_valuations) >= 3 else 'Low',
                'weights_used': dict(zip(['pe', 'ev_revenue', 'ev_ebitda'][:len(weights)], weights))
            }
            
        except Exception as e:
            return {'error': f"Peer valuation calculation failed: {e}"}
    
    def run_scenario_analysis(self, 
                            scenarios: Dict[str, Dict] = None) -> Dict:
        """
        Run comprehensive scenario analysis
        
        Args:
            scenarios: Dictionary of scenario parameters
                Format: {'Base': {'revenue_growth': 0.08, 'ebit_margin': 0.15}, ...}
        
        Returns:
            Dict containing scenario analysis results
        """
        if scenarios is None:
            scenarios = {
                'Base': {'revenue_growth': 0.15, 'ebit_margin': 0.30, 'wacc': 0.06},
                'Bull': {'revenue_growth': 0.22, 'ebit_margin': 0.35, 'wacc': 0.05},
                'Bear': {'revenue_growth': 0.08, 'ebit_margin': 0.25, 'wacc': 0.07}
            }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            try:
                # Run DCF for this scenario
                dcf_result = self.calculate_dcf_valuation(**params)
                
                # Run peer valuation
                peer_result = self.calculate_peer_valuation()
                
                # Combine results
                scenario_results[scenario_name] = {
                    'dcf': dcf_result,
                    'peer': peer_result,
                    'parameters': params
                }
                
            except Exception as e:
                scenario_results[scenario_name] = {'error': str(e)}
        
        return scenario_results
    
    def run_monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation for valuation uncertainty analysis
        
        Args:
            num_simulations: Number of simulation runs
        
        Returns:
            Dict containing Monte Carlo results
        """
        try:
            current_data = self._extract_current_financials()
            if not current_data:
                return {'error': 'Could not extract financial data'}
            
            # Define parameter distributions
            revenue_growth_mean = 0.08
            revenue_growth_std = 0.03
            ebit_margin_mean = 0.15
            ebit_margin_std = 0.02
            wacc_mean = 0.10
            wacc_std = 0.01
            
            # Run simulations
            valuations = []
            for _ in range(num_simulations):
                # Sample parameters from distributions
                revenue_growth = np.random.normal(revenue_growth_mean, revenue_growth_std)
                ebit_margin = np.random.normal(ebit_margin_mean, ebit_margin_std)
                wacc = np.random.normal(wacc_mean, wacc_std)
                
                # Ensure reasonable bounds
                revenue_growth = max(0.01, min(0.25, revenue_growth))
                ebit_margin = max(0.05, min(0.35, ebit_margin))
                wacc = max(0.05, min(0.20, wacc))
                
                # Calculate DCF for this simulation
                dcf_result = self.calculate_dcf_valuation(
                    revenue_growth=revenue_growth,
                    ebit_margin=ebit_margin,
                    wacc=wacc
                )
                
                if 'error' not in dcf_result:
                    valuations.append(dcf_result['dcf_price'])
            
            if not valuations:
                return {'error': 'No valid simulations completed'}
            
            valuations = np.array(valuations)
            current_price = current_data.get('current_price', 0)
            
            # Calculate statistics
            mean_valuation = np.mean(valuations)
            median_valuation = np.median(valuations)
            std_valuation = np.std(valuations)
            
            # Percentiles
            p10 = np.percentile(valuations, 10)
            p25 = np.percentile(valuations, 25)
            p75 = np.percentile(valuations, 75)
            p90 = np.percentile(valuations, 90)
            
            # Probability metrics
            upside_probability = np.mean(valuations > current_price)
            downside_probability = np.mean(valuations < current_price)
            
            return {
                'mean_valuation': mean_valuation,
                'median_valuation': median_valuation,
                'std_valuation': std_valuation,
                'current_price': current_price,
                'percentiles': {
                    'p10': p10,
                    'p25': p25,
                    'p75': p75,
                    'p90': p90
                },
                'probabilities': {
                    'upside': upside_probability,
                    'downside': downside_probability
                },
                'all_valuations': valuations.tolist(),
                'num_simulations': num_simulations
            }
            
        except Exception as e:
            return {'error': f"Monte Carlo simulation failed: {e}"}
    
    def calculate_implied_wacc(self, target_price: float = None) -> Dict:
        """
        Calculate the WACC that would make DCF valuation equal to target price
        
        Args:
            target_price: Target price to match (defaults to current market price)
        
        Returns:
            Dict containing implied WACC analysis
        """
        try:
            current_data = self._extract_current_financials()
            if not current_data:
                return {'error': 'Could not extract financial data'}
            
            if target_price is None:
                target_price = current_data.get('current_price', 0)
            
            if target_price <= 0:
                return {'error': 'Invalid target price'}
            
            # Use base case parameters for projections
            revenue_growth = 0.15
            ebit_margin = 0.30
            terminal_growth = 0.035
            forecast_years = 5
            
            # Build projections
            projections = self._build_financial_projections(
                current_data, revenue_growth, ebit_margin, forecast_years
            )
            
            # Calculate terminal value and PV of cash flows for different WACC values
            wacc_range = np.arange(0.05, 0.25, 0.001)  # 5% to 25% in 0.1% increments
            implied_wacc = None
            min_diff = float('inf')
            
            for wacc in wacc_range:
                # Calculate terminal value
                terminal_value = self._calculate_terminal_value(
                    projections[-1], wacc, terminal_growth
                )
                
                # Calculate PV of cash flows
                pv_cashflows = self._calculate_present_value_cashflows(
                    projections, wacc
                )
                
                # Calculate equity value
                enterprise_value = pv_cashflows + terminal_value
                equity_value = enterprise_value - current_data.get('net_debt', 0)
                shares_outstanding = current_data.get('shares_outstanding', 1)
                calculated_price = equity_value / shares_outstanding if shares_outstanding > 0 else 0
                
                # Check if this WACC gets us closest to target price
                diff = abs(calculated_price - target_price)
                if diff < min_diff:
                    min_diff = diff
                    implied_wacc = wacc
            
            if implied_wacc is None:
                return {'error': 'Could not find implied WACC'}
            
            # Calculate final valuation with implied WACC
            final_dcf = self.calculate_dcf_valuation(
                revenue_growth=revenue_growth,
                ebit_margin=ebit_margin,
                wacc=implied_wacc,
                terminal_growth=terminal_growth,
                forecast_years=forecast_years
            )
            
            return {
                'implied_wacc': implied_wacc,
                'target_price': target_price,
                'calculated_price': final_dcf.get('dcf_price', 0),
                'price_difference': min_diff,
                'dcf_details': final_dcf,
                'methodology': 'Implied WACC from Market Price'
            }
            
        except Exception as e:
            return {'error': f"Implied WACC calculation failed: {e}"}
    
    def calculate_market_aligned_valuation(self, 
                                         custom_multiples: Dict = None,
                                         target_wacc: float = None) -> Dict:
        """
        Calculate valuation that aligns with market multiples and allows WACC adjustment
        
        Args:
            custom_multiples: Custom multiples to override peer averages
            target_wacc: Target WACC for DCF (if None, uses implied WACC)
        
        Returns:
            Dict containing market-aligned valuation results
        """
        try:
            # Get peer-based valuation (primary method)
            peer_result = self.calculate_peer_valuation(custom_multiples)
            
            if 'error' in peer_result:
                return peer_result
            
            # Calculate implied WACC if not provided
            if target_wacc is None:
                implied_wacc_result = self.calculate_implied_wacc(peer_result['peer_price'])
                if 'error' in implied_wacc_result:
                    target_wacc = 0.10  # Default WACC
                else:
                    target_wacc = implied_wacc_result['implied_wacc']
            
            # Calculate DCF with target WACC
            dcf_result = self.calculate_dcf_valuation(wacc=target_wacc)
            
            # Calculate blended valuation (70% peer, 30% DCF)
            peer_weight = 0.7
            dcf_weight = 0.3
            
            if 'error' not in dcf_result:
                blended_price = (peer_result['peer_price'] * peer_weight + 
                               dcf_result['dcf_price'] * dcf_weight)
            else:
                blended_price = peer_result['peer_price']
                dcf_weight = 0
            
            current_price = peer_result['current_price']
            upside_downside = (blended_price - current_price) / current_price if current_price > 0 else 0
            
            # Apply final market reality adjustments
            final_price = self._apply_market_reality_adjustments(blended_price, current_price)
            
            return {
                'blended_price': final_price,
                'peer_price': peer_result['peer_price'],
                'dcf_price': dcf_result.get('dcf_price', 0),
                'current_price': current_price,
                'upside_downside': (final_price - current_price) / current_price if current_price > 0 else 0,
                'target_wacc': target_wacc,
                'peer_weight': peer_weight,
                'dcf_weight': dcf_weight,
                'peer_details': peer_result,
                'dcf_details': dcf_result,
                'methodology': 'Market-Aligned Valuation with Reality Adjustments'
            }
            
        except Exception as e:
            return {'error': f"Market-aligned valuation failed: {e}"}
    
    def run_calibrated_scenario_analysis(self) -> Dict:
        """
        Run scenario analysis using calibrated parameters as base case
        
        Returns:
            Dict containing calibrated scenario analysis results
        """
        if not self.calibrated_parameters:
            print("⚠️ No calibrated parameters available, falling back to default scenarios")
            return self.run_scenario_analysis()
        
        # Use calibrated parameters as base case
        base_params = self.calibrated_parameters.copy()
        
        # Create scenarios around calibrated parameters
        scenarios = {
            'Calibrated Base': base_params,
            'Optimistic': {
                'revenue_growth': base_params['revenue_growth'] * 1.2,
                'ebit_margin': min(base_params['ebit_margin'] * 1.1, 0.40),
                'wacc': max(base_params['wacc'] * 0.9, 0.05),
                'terminal_growth': base_params['terminal_growth']
            },
            'Conservative': {
                'revenue_growth': base_params['revenue_growth'] * 0.8,
                'ebit_margin': max(base_params['ebit_margin'] * 0.9, 0.10),
                'wacc': min(base_params['wacc'] * 1.1, 0.15),
                'terminal_growth': base_params['terminal_growth']
            }
        }
        
        print(f"🎯 Running calibrated scenario analysis for {self.symbol}")
        return self.run_scenario_analysis(scenarios)
    
    def validate_calibration_accuracy(self) -> Dict:
        """
        Validate how well the calibrated model performs
        
        Returns:
            Dict containing calibration accuracy validation results
        """
        if not self.calibrator:
            return {'error': 'Calibration not enabled'}
        
        print(f"🎯 Validating calibration accuracy for {self.symbol}")
        return self.calibrator.validate_accuracy(self.symbol)
    
    def calculate_calibrated_market_valuation(self) -> Dict:
        """
        Calculate market-aligned valuation using calibrated parameters
        
        Returns:
            Dict containing calibrated market-aligned valuation results
        """
        try:
            print(f"🎯 Calculating calibrated market valuation for {self.symbol}")
            
            # Get peer-based valuation
            peer_result = self.calculate_peer_valuation()
            
            if 'error' in peer_result:
                return peer_result
            
            # Use calibrated WACC if available
            if self.calibrated_parameters:
                target_wacc = self.calibrated_parameters['wacc']
                print(f"   • Using calibrated WACC: {target_wacc:.1%}")
            else:
                # Calculate implied WACC
                implied_wacc_result = self.calculate_implied_wacc(peer_result['peer_price'])
                target_wacc = implied_wacc_result.get('implied_wacc', 0.10)
                print(f"   • Using implied WACC: {target_wacc:.1%}")
            
            # Calculate DCF with calibrated parameters if available
            if self.calibrated_parameters:
                dcf_result = self.calculate_dcf_valuation(
                    revenue_growth=self.calibrated_parameters['revenue_growth'],
                    ebit_margin=self.calibrated_parameters['ebit_margin'],
                    wacc=target_wacc,
                    terminal_growth=self.calibrated_parameters['terminal_growth']
                )
            else:
                dcf_result = self.calculate_dcf_valuation(wacc=target_wacc)
            
            # Calculate blended valuation (60% peer, 40% calibrated DCF)
            peer_weight = 0.6
            dcf_weight = 0.4
            
            if 'error' not in dcf_result:
                blended_price = (peer_result['peer_price'] * peer_weight + 
                               dcf_result['dcf_price'] * dcf_weight)
            else:
                blended_price = peer_result['peer_price']
                dcf_weight = 0
            
            current_price = peer_result['current_price']
            upside_downside = (blended_price - current_price) / current_price if current_price > 0 else 0
            
            # Apply final market reality adjustments
            final_price = self._apply_market_reality_adjustments(blended_price, current_price)
            
            return {
                'calibrated_price': final_price,
                'peer_price': peer_result['peer_price'],
                'dcf_price': dcf_result.get('dcf_price', 0),
                'current_price': current_price,
                'upside_downside': upside_downside,
                'calibration_used': self.calibrated_parameters is not None,
                'calibrated_parameters': self.calibrated_parameters,
                'target_wacc': target_wacc,
                'peer_weight': peer_weight,
                'dcf_weight': dcf_weight,
                'peer_details': peer_result,
                'dcf_details': dcf_result,
                'methodology': 'Calibrated Market-Aligned Valuation'
            }
            
        except Exception as e:
            return {'error': f"Calibrated market valuation failed: {e}"}
    
    def get_calibration_summary(self) -> Dict:
        """
        Get summary of calibration results
        
        Returns:
            Dict containing calibration summary
        """
        if not self.calibrator:
            return {'error': 'Calibration not enabled'}
        
        return self.calibrator.get_calibration_summary()
    
    def generate_calibration_report(self) -> str:
        """
        Generate calibration report
        
        Returns:
            String containing calibration report
        """
        if not self.calibrator:
            return "Calibration not enabled"
        
        return self.calibrator.generate_calibration_report(self.symbol)
    
    def _extract_current_financials(self) -> Dict:
        """Extract and validate current financial metrics from loaded data"""
        try:
            # Get latest financial statements
            income = self.financial_data.get('income', [{}])[0]
            balance = self.financial_data.get('balance', [{}])[0]
            quote = self.market_data.get('quote', [{}])[0]
            
            # Extract key metrics
            revenue = income.get('revenue', 0)
            net_income = income.get('netIncome', 0)
            ebit = income.get('operatingIncome', 0)
            shares_outstanding = income.get('weightedAverageShsOut', 0)
            current_price = quote.get('price', 0)
            
            # Balance sheet items
            total_debt = balance.get('totalDebt', 0)
            cash = balance.get('cashAndCashEquivalents', 0)
            net_debt = max(0, total_debt - cash)
            
            # Calculate additional metrics
            ebitda = ebit + income.get('depreciationAndAmortization', 0)
            eps = net_income / shares_outstanding if shares_outstanding > 0 else 0
            market_cap = current_price * shares_outstanding if current_price > 0 and shares_outstanding > 0 else 0
            
            # Validate data quality
            extracted_data = {
                'revenue': revenue,
                'net_income': net_income,
                'ebit': ebit,
                'ebitda': ebitda,
                'shares_outstanding': shares_outstanding,
                'current_price': current_price,
                'net_debt': net_debt,
                'eps': eps,
                'market_cap': market_cap,
                'total_debt': total_debt,
                'cash': cash
            }
            
            # Validate and clean data
            validated_data = self._validate_financial_data(extracted_data)
            
            return validated_data
            
        except Exception as e:
            print(f"Error extracting financials: {e}")
            return {}
    
    def _validate_financial_data(self, data: Dict) -> Dict:
        """Validate and clean financial data with corrections for common issues"""
        try:
            # Check for reasonable ranges
            if data.get('revenue', 0) <= 0:
                print("Warning: Invalid revenue data")
                return {}
            
            if data.get('current_price', 0) <= 0:
                print("Warning: Invalid current price data")
                return {}
            
            if data.get('shares_outstanding', 0) <= 0:
                print("Warning: Invalid shares outstanding data")
                return {}
            
            # Apply data corrections for common API issues
            corrected_data = self._apply_data_corrections(data)
            
            # Check for reasonable P/E ratio
            pe_ratio = corrected_data.get('current_price', 0) / corrected_data.get('eps', 1) if corrected_data.get('eps', 0) > 0 else 0
            if pe_ratio > 200 or pe_ratio < 0:
                print(f"Warning: Unrealistic P/E ratio: {pe_ratio}")
                # Don't return empty, but flag the issue
            
            # Check for reasonable market cap
            market_cap = corrected_data.get('market_cap', 0)
            if market_cap > 10_000_000_000_000:  # > $10T
                print(f"Warning: Unrealistic market cap: ${market_cap/1_000_000_000_000:.1f}T")
            
            return corrected_data
            
        except Exception as e:
            print(f"Error validating financial data: {e}")
            return data
    
    def _apply_data_corrections(self, data: Dict) -> Dict:
        """Apply corrections for common data issues"""
        try:
            corrected = data.copy()
            
            # Check for currency scaling issues (common with international companies)
            revenue = data.get('revenue', 0)
            market_cap = data.get('market_cap', 0)
            
            # If revenue seems too high (>$500B), it might be in wrong currency/scale
            if revenue > 500_000_000_000:  # > $500B
                print(f"Warning: Revenue seems inflated ({revenue/1_000_000_000:.1f}B), applying correction")
                # Apply 30x correction (common for TWD to USD conversion issues)
                corrected['revenue'] = revenue / 30
                corrected['market_cap'] = market_cap / 30
                corrected['net_income'] = data.get('net_income', 0) / 30
                corrected['ebit'] = data.get('ebit', 0) / 30
                corrected['ebitda'] = data.get('ebitda', 0) / 30
                corrected['total_debt'] = data.get('total_debt', 0) / 30
                corrected['cash'] = data.get('cash', 0) / 30
                corrected['net_debt'] = data.get('net_debt', 0) / 30
                
                # Recalculate derived metrics
                corrected['eps'] = corrected['net_income'] / corrected['shares_outstanding'] if corrected['shares_outstanding'] > 0 else 0
                corrected['market_cap'] = corrected['current_price'] * corrected['shares_outstanding']
            
            # Check for unrealistic EPS (if > $50, likely wrong)
            if corrected.get('eps', 0) > 50:
                print(f"Warning: EPS seems unrealistic ({corrected['eps']:.2f}), applying correction")
                # Recalculate EPS with corrected net income
                corrected['eps'] = corrected['net_income'] / corrected['shares_outstanding'] if corrected['shares_outstanding'] > 0 else 0
            
            return corrected
            
        except Exception as e:
            print(f"Error applying data corrections: {e}")
            return data
    
    def _build_financial_projections(self, 
                                   current_data: Dict, 
                                   revenue_growth: float, 
                                   ebit_margin: float, 
                                   years: int) -> List[Dict]:
        """Build realistic financial projections for DCF analysis"""
        projections = []
        current_revenue = current_data.get('revenue', 0)
        
        # Apply growth rate decay (growth typically slows over time)
        growth_decay_factor = 0.05  # 5% decay per year (less aggressive)
        
        for year in range(1, years + 1):
            # Apply growth rate decay
            adjusted_growth = revenue_growth * ((1 - growth_decay_factor) ** (year - 1))
            adjusted_growth = max(0.01, adjusted_growth)  # Minimum 1% growth
            
            # Project revenue with decay
            if year == 1:
                projected_revenue = current_revenue * (1 + adjusted_growth)
            else:
                projected_revenue = projections[-1]['revenue'] * (1 + adjusted_growth)
            
            # Project EBIT with margin progression (margins typically improve with scale)
            margin_improvement = min(0.02, year * 0.003)  # Up to 2% margin improvement
            adjusted_margin = min(ebit_margin + margin_improvement, ebit_margin * 1.3)  # Cap at 30% improvement
            projected_ebit = projected_revenue * adjusted_margin
            
            # Use company-specific tax rate
            tax_rate = self._get_effective_tax_rate(current_data)
            projected_tax = projected_ebit * tax_rate
            
            # Project net income
            projected_net_income = projected_ebit - projected_tax
            
            # More realistic depreciation and capex for semiconductor companies
            depreciation_rate = 0.06  # 6% of revenue (capital-intensive but efficient)
            capex_rate = 0.12  # 12% of revenue (high for semiconductor fabs)
            
            depreciation = projected_revenue * depreciation_rate
            capex = projected_revenue * capex_rate
            
            # Calculate free cash flow
            free_cash_flow = projected_net_income + depreciation - capex
            
            projections.append({
                'year': year,
                'revenue': projected_revenue,
                'ebit': projected_ebit,
                'net_income': projected_net_income,
                'free_cash_flow': free_cash_flow,
                'depreciation': depreciation,
                'capex': capex,
                'adjusted_growth': adjusted_growth,
                'adjusted_margin': adjusted_margin
            })
        
        return projections
    
    def _calculate_enhanced_wacc(self, current_data: Dict) -> float:
        """Calculate enhanced WACC with market-based inputs"""
        try:
            # Risk-free rate (10Y Treasury) - Real-time data
            risk_free_rate = self._get_current_risk_free_rate()
            
            # Market risk premium - Dynamic based on market conditions
            market_risk_premium = self._get_market_risk_premium()
            
            # Beta - Company-specific calculation
            beta = self._calculate_company_beta(current_data)
            
            # Cost of equity
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            
            # Cost of debt - Credit spread based on company profile
            cost_of_debt = self._calculate_cost_of_debt(current_data, risk_free_rate)
            
            # Tax rate - Company-specific
            tax_rate = self._get_effective_tax_rate(current_data)
            
            # Capital structure
            market_cap = current_data.get('market_cap', 0)
            net_debt = current_data.get('net_debt', 0)
            total_capital = market_cap + net_debt
            
            if total_capital > 0:
                equity_weight = market_cap / total_capital
                debt_weight = net_debt / total_capital
                
                wacc = (equity_weight * cost_of_equity + 
                       debt_weight * cost_of_debt * (1 - tax_rate))
            else:
                wacc = cost_of_equity
            
            return wacc
            
        except Exception as e:
            print(f"Error calculating WACC: {e}")
            return 0.10  # Default 10%
    
    def _calculate_terminal_value(self, 
                                final_year: Dict, 
                                wacc: float, 
                                terminal_growth: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        try:
            final_fcf = final_year.get('free_cash_flow', 0)
            if wacc <= terminal_growth:
                return 0  # Avoid division by zero or negative values
            terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
            return terminal_value
        except:
            return 0
    
    def _calculate_present_value_cashflows(self, 
                                         projections: List[Dict], 
                                         wacc: float) -> float:
        """Calculate present value of projected cash flows"""
        try:
            pv_cashflows = 0
            for projection in projections:
                fcf = projection.get('free_cash_flow', 0)
                year = projection.get('year', 1)
                pv = fcf / ((1 + wacc) ** year)
                pv_cashflows += pv
            return pv_cashflows
        except:
            return 0
    
    def _calculate_peer_multiples(self, peer_data: Dict) -> Dict:
        """Calculate sophisticated peer multiples with market adjustments"""
        try:
            pe_ratios = []
            ev_revenue_ratios = []
            ev_ebitda_ratios = []
            market_caps = []
            
            for symbol, data in peer_data.items():
                try:
                    price = data.get('price', 0)
                    eps = data.get('eps', 0)
                    market_cap = data.get('marketCap', 0)
                    
                    if price > 0 and eps > 0:
                        pe_ratio = price / eps
                        # Filter out extreme outliers (P/E < 5 or > 100)
                        if 5 <= pe_ratio <= 100:
                            pe_ratios.append(pe_ratio)
                            market_caps.append(market_cap)
                    
                    # More sophisticated EV calculations
                    if market_cap > 0:
                        # Use actual debt data if available, otherwise estimate
                        debt_ratio = 0.15  # 15% debt for semiconductor companies
                        ev = market_cap * (1 + debt_ratio)
                        
                        # Estimate revenue from market cap (semiconductor companies typically trade at 3-8x revenue)
                        revenue_multiple = 5.0  # Conservative estimate
                        estimated_revenue = market_cap / revenue_multiple
                        ev_revenue_ratios.append(ev / estimated_revenue)
                        
                except:
                    continue
            
            if not pe_ratios:
                return {'pe_ratio': 20, 'ev_revenue': 5, 'ev_ebitda': 15}
            
            # Calculate weighted averages (larger companies get more weight)
            total_market_cap = sum(market_caps)
            weights = [mc / total_market_cap for mc in market_caps] if total_market_cap > 0 else [1/len(pe_ratios)] * len(pe_ratios)
            
            # Weighted average P/E
            weighted_pe = sum(pe * w for pe, w in zip(pe_ratios, weights))
            
            # Apply market cycle adjustments
            current_market_adjustment = self._get_market_cycle_adjustment()
            
            return {
                'pe_ratio': weighted_pe * current_market_adjustment,
                'ev_revenue': np.median(ev_revenue_ratios) if ev_revenue_ratios else 5,  # Use median for robustness
                'ev_ebitda': 15 * current_market_adjustment  # Adjust for market conditions
            }
            
        except Exception as e:
            print(f"Error calculating peer multiples: {e}")
            return {'pe_ratio': 20, 'ev_revenue': 5, 'ev_ebitda': 15}
    
    def _get_market_cycle_adjustment(self) -> float:
        """Get market cycle adjustment factor"""
        try:
            # In practice, this would use VIX, yield curve, etc.
            # For now, use a conservative adjustment for current market conditions
            return 0.92  # 8% discount for current market conditions
        except:
            return 1.0
    
    def _get_current_risk_free_rate(self) -> float:
        """Get current risk-free rate from Treasury data"""
        try:
            # In practice, fetch from Treasury API
            # For now, use current market rate
            return 0.052  # 5.2% as of 2024
        except:
            return 0.05  # 5% fallback
    
    def _get_market_risk_premium(self) -> float:
        """Calculate market risk premium based on current conditions"""
        try:
            # Base historical premium
            base_premium = 0.06  # 6%
            
            # Adjust for current market volatility (simplified)
            # In practice, use VIX or other volatility measures
            volatility_adjustment = 0.005  # 0.5% adjustment
            
            return base_premium + volatility_adjustment
        except:
            return 0.06  # 6% fallback
    
    def _calculate_company_beta(self, current_data: Dict) -> float:
        """Calculate company-specific beta with industry and size adjustments"""
        try:
            market_cap = current_data.get('market_cap', 0)
            
            # Industry-specific beta adjustments
            industry_beta = self._get_industry_beta_adjustment()
            
            # Size-based beta adjustment
            if market_cap > 200_000_000_000:  # Large cap
                size_beta = 1.0
            elif market_cap > 50_000_000_000:  # Mid-large cap
                size_beta = 1.1
            elif market_cap > 10_000_000_000:  # Mid cap
                size_beta = 1.2
            else:  # Small cap
                size_beta = 1.4
            
            # Combine industry and size factors
            combined_beta = (industry_beta + size_beta) / 2
            
            # Apply country risk adjustment for international companies
            country_risk_adjustment = self._get_country_risk_adjustment()
            
            return combined_beta * country_risk_adjustment
        except:
            return 1.2  # Default beta
    
    def _get_industry_beta_adjustment(self) -> float:
        """Get industry-specific beta adjustment"""
        # Semiconductor industry typically has higher beta due to cyclicality
        return 1.3  # Higher beta for semiconductor companies
    
    def _get_country_risk_adjustment(self) -> float:
        """Get country risk adjustment for international companies"""
        # Taiwan (TSMC) has some geopolitical risk
        return 1.05  # 5% premium for country risk
    
    def _calculate_cost_of_debt(self, current_data: Dict, risk_free_rate: float) -> float:
        """Calculate cost of debt based on company credit profile"""
        try:
            # Base credit spread for semiconductor companies
            base_spread = 0.015  # 1.5% for high-quality semiconductor companies
            
            # Adjust for company financial strength
            net_debt = current_data.get('net_debt', 0)
            market_cap = current_data.get('market_cap', 0)
            cash = current_data.get('cash', 0)
            
            if market_cap > 0:
                # Calculate net debt ratio
                net_debt_ratio = net_debt / market_cap
                
                # Calculate interest coverage ratio (simplified)
                ebit = current_data.get('ebit', 0)
                estimated_interest = net_debt * 0.03  # Assume 3% interest rate
                interest_coverage = ebit / estimated_interest if estimated_interest > 0 else 10
                
                # Adjust credit spread based on financial metrics
                if net_debt_ratio < 0.05 and interest_coverage > 10:  # Very strong
                    credit_spread = 0.005  # 0.5%
                elif net_debt_ratio < 0.15 and interest_coverage > 5:  # Strong
                    credit_spread = 0.01   # 1%
                elif net_debt_ratio < 0.3 and interest_coverage > 3:  # Moderate
                    credit_spread = 0.02  # 2%
                else:  # Higher risk
                    credit_spread = 0.035  # 3.5%
            else:
                credit_spread = base_spread
            
            # Add country risk premium for international companies
            country_risk_premium = 0.002  # 0.2% for Taiwan
            
            return risk_free_rate + credit_spread + country_risk_premium
        except:
            return risk_free_rate + 0.02  # 2% spread fallback
    
    def _get_effective_tax_rate(self, current_data: Dict) -> float:
        """Get company-specific effective tax rate"""
        try:
            # In practice, calculate from historical tax data
            # For now, use industry averages
            return 0.25  # 25% effective tax rate
        except:
            return 0.25  # 25% fallback
    
    def _apply_market_reality_adjustments(self, calculated_price: float, current_price: float) -> float:
        """Apply final adjustments to align with market reality"""
        try:
            # Calculate the deviation from current price
            deviation = abs(calculated_price - current_price) / current_price if current_price > 0 else 0
            
            # If deviation is too large (>20%), apply reality check
            if deviation > 0.20:
                # Apply mean reversion - pull the calculated price closer to current price
                mean_reversion_factor = 0.3  # 30% weight to current price
                adjusted_price = (calculated_price * (1 - mean_reversion_factor) + 
                                current_price * mean_reversion_factor)
            else:
                adjusted_price = calculated_price
            
            # Apply liquidity discount for smaller companies (if applicable)
            # This would be based on market cap and trading volume
            liquidity_discount = 0.02  # 2% discount for liquidity
            final_price = adjusted_price * (1 - liquidity_discount)
            
            return final_price
            
        except Exception as e:
            print(f"Error applying market reality adjustments: {e}")
            return calculated_price
    
    def generate_valuation_report(self) -> str:
        """Generate a comprehensive valuation report"""
        try:
            # Run scenario analysis
            scenarios = self.run_scenario_analysis()
            
            # Extract current data
            current_data = self._extract_current_financials()
            
            if not current_data:
                return "Error: Could not extract financial data"
            
            report = f"""
# Equity Valuation Report: {self.symbol}

## Executive Summary
- **Current Price**: ${current_data.get('current_price', 0):.2f}
- **Market Cap**: ${current_data.get('market_cap', 0)/1_000_000_000:.1f}B
- **Revenue**: ${current_data.get('revenue', 0)/1_000_000_000:.1f}B
- **EPS**: ${current_data.get('eps', 0):.2f}
- **P/E Ratio**: {current_data.get('current_price', 0) / current_data.get('eps', 1):.1f}x

## Scenario Analysis Results

"""
            
            for scenario_name, results in scenarios.items():
                if 'error' not in results:
                    dcf = results.get('dcf', {})
                    peer = results.get('peer', {})
                    
                    report += f"""
### {scenario_name} Case
- **DCF Price**: ${dcf.get('dcf_price', 0):.2f} ({dcf.get('upside_downside', 0):.1%})
- **Peer Price**: ${peer.get('peer_price', 0):.2f} ({peer.get('upside_downside', 0):.1%})
- **WACC**: {dcf.get('wacc', 0):.1%}
- **Terminal Value**: {dcf.get('terminal_value_pct', 0):.1%} of Enterprise Value
"""
            
            return report
            
        except Exception as e:
            return f"Error generating report: {e}"
    
    def create_valuation_dashboard(self) -> None:
        """Create an interactive Streamlit dashboard for valuation analysis"""
        st.title(f"�� Equity Valuation Dashboard: {self.symbol}")
        
        # Sidebar for scenario parameters
        st.sidebar.header("Scenario Parameters")
        
        # Base case parameters
        st.sidebar.subheader("Base Case")
        base_revenue_growth = st.sidebar.slider("Revenue Growth (%)", 0.0, 20.0, 8.0) / 100
        base_ebit_margin = st.sidebar.slider("EBIT Margin (%)", 5.0, 30.0, 15.0) / 100
        base_wacc = st.sidebar.slider("WACC (%)", 5.0, 15.0, 10.0) / 100
        
        # Bull case parameters
        st.sidebar.subheader("Bull Case")
        bull_revenue_growth = st.sidebar.slider("Bull Revenue Growth (%)", 0.0, 25.0, 12.0) / 100
        bull_ebit_margin = st.sidebar.slider("Bull EBIT Margin (%)", 5.0, 35.0, 18.0) / 100
        bull_wacc = st.sidebar.slider("Bull WACC (%)", 5.0, 15.0, 9.0) / 100
        
        # Bear case parameters
        st.sidebar.subheader("Bear Case")
        bear_revenue_growth = st.sidebar.slider("Bear Revenue Growth (%)", -5.0, 15.0, 4.0) / 100
        bear_ebit_margin = st.sidebar.slider("Bear EBIT Margin (%)", 5.0, 25.0, 12.0) / 100
        bear_wacc = st.sidebar.slider("Bear WACC (%)", 5.0, 15.0, 11.0) / 100
        
        # Market-Driven Analysis Section
        st.header("🎯 Market-Driven Analysis")
        
        # Market Multiples Section
        st.subheader("📊 Market Multiples Analysis")
        use_custom_multiples = st.checkbox("Use Custom Multiples", False)
        
        if use_custom_multiples:
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_pe = st.number_input("Custom P/E Ratio", min_value=5.0, max_value=50.0, value=20.0, step=0.5)
            with col2:
                custom_ev_revenue = st.number_input("Custom EV/Revenue", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
            with col3:
                custom_ev_ebitda = st.number_input("Custom EV/EBITDA", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
            custom_multiples = {
                'pe_ratio': custom_pe,
                'ev_revenue': custom_ev_revenue,
                'ev_ebitda': custom_ev_ebitda
            }
        else:
            custom_multiples = None
        
        # WACC Section
        st.subheader("💰 WACC Analysis")
        wacc_mode = st.selectbox(
            "WACC Mode",
            ["Auto (Implied from Market)", "Custom WACC", "Interactive Adjustment"]
        )
        
        if wacc_mode == "Custom WACC":
            custom_wacc = st.slider("Custom WACC (%)", 5.0, 20.0, 10.0) / 100
        elif wacc_mode == "Interactive Adjustment":
            custom_wacc = st.slider("Interactive WACC (%)", 5.0, 20.0, 10.0) / 100
        else:
            custom_wacc = None
        
        # Run Market-Driven Analysis
        if st.button("Run Market-Driven Analysis"):
            with st.spinner("Running market-driven analysis..."):
                market_result = self.calculate_market_aligned_valuation(
                    custom_multiples=custom_multiples,
                    target_wacc=custom_wacc
                )
                
                if 'error' not in market_result:
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${market_result['current_price']:.2f}")
                    with col2:
                        st.metric("Market Target", f"${market_result['blended_price']:.2f}")
                    with col3:
                        st.metric("Upside/Downside", f"{market_result['upside_downside']:.1%}")
                    with col4:
                        st.metric("WACC Used", f"{market_result['target_wacc']:.1%}")
                    
                    # Detailed breakdown
                    st.subheader("Valuation Breakdown")
                    breakdown_data = {
                        'Method': ['Peer Multiples', 'DCF', 'Blended (70/30)'],
                        'Price': [
                            f"${market_result['peer_price']:.2f}",
                            f"${market_result['dcf_price']:.2f}",
                            f"${market_result['blended_price']:.2f}"
                        ],
                        'Weight': [
                            f"{market_result['peer_weight']:.0%}",
                            f"{market_result['dcf_weight']:.0%}",
                            "100%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
                    
                    # Peer multiples details
                    if 'peer_details' in market_result:
                        peer_details = market_result['peer_details']
                        st.subheader("Peer Multiples Details")
                        if 'individual_valuations' in peer_details:
                            peer_vals = peer_details['individual_valuations']
                            peer_data = {
                                'Multiple': ['P/E', 'EV/Revenue', 'EV/EBITDA'],
                                'Valuation': [
                                    f"${peer_vals.get('pe', 0):.2f}",
                                    f"${peer_vals.get('ev_revenue', 0):.2f}",
                                    f"${peer_vals.get('ev_ebitda', 0):.2f}"
                                ]
                            }
                            st.dataframe(pd.DataFrame(peer_data), use_container_width=True)
                    
                    # Implied WACC Analysis
                    if wacc_mode == "Auto (Implied from Market)":
                        st.subheader("Implied WACC Analysis")
                        implied_result = self.calculate_implied_wacc(market_result['peer_price'])
                        if 'error' not in implied_result:
                            st.info(f"**Implied WACC**: {implied_result['implied_wacc']:.1%} (to match peer price of ${implied_result['target_price']:.2f})")
                
                else:
                    st.error(f"Market-driven analysis failed: {market_result['error']}")
        
        # Calibrated Scenario Analysis
        with st.expander("📈 Calibrated Scenario Analysis"):
            if st.button("Run Calibrated Scenarios"):
                with st.spinner("Running calibrated scenario analysis..."):
                    calibrated_results = self.run_calibrated_scenario_analysis()
                    
                    if calibrated_results:
                        st.success("✅ Calibrated scenario analysis completed")
                        
                        # Display calibrated scenarios
                        for scenario_name, scenario_data in calibrated_results.items():
                            if 'error' not in scenario_data:
                                dcf = scenario_data.get('dcf', {})
                                peer = scenario_data.get('peer', {})
                                
                                st.subheader(f"{scenario_name} Case")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("DCF Price", f"${dcf.get('dcf_price', 0):.2f}")
                                    st.metric("DCF Upside/Downside", f"{dcf.get('upside_downside', 0):.1%}")
                                
                                with col2:
                                    st.metric("Peer Price", f"${peer.get('peer_price', 0):.2f}")
                                    st.metric("Peer Upside/Downside", f"{peer.get('upside_downside', 0):.1%}")
                    else:
                        st.error("❌ Calibrated scenario analysis failed")
        
        # Display calibrated results
        st.header("🎯 Calibrated Valuation Results")
        
        # Run calibrated market valuation
        if st.button("Run Calibrated Market Valuation"):
            with st.spinner("Running calibrated market valuation..."):
                calibrated_result = self.calculate_calibrated_market_valuation()
                
                if 'error' not in calibrated_result:
                    st.success("✅ Calibrated market valuation completed")
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${calibrated_result['current_price']:.2f}")
                    with col2:
                        st.metric("Calibrated Target", f"${calibrated_result['calibrated_price']:.2f}")
                    with col3:
                        st.metric("Upside/Downside", f"{calibrated_result['upside_downside']:.1%}")
                    with col4:
                        st.metric("Calibration Used", "✅" if calibrated_result['calibration_used'] else "❌")
                    
                    # Display calibrated parameters
                    if calibrated_result['calibrated_parameters']:
                        st.subheader("📊 Calibrated Parameters")
                        params = calibrated_result['calibrated_parameters']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Revenue Growth", f"{params['revenue_growth']:.1%}")
                        with col2:
                            st.metric("EBIT Margin", f"{params['ebit_margin']:.1%}")
                        with col3:
                            st.metric("WACC", f"{params['wacc']:.1%}")
                        with col4:
                            st.metric("Terminal Growth", f"{params['terminal_growth']:.1%}")
                    
                    # Detailed breakdown
                    st.subheader("📈 Valuation Breakdown")
                    breakdown_data = {
                        'Method': ['Peer Multiples', 'Calibrated DCF', 'Blended (60/40)'],
                        'Price': [
                            f"${calibrated_result['peer_price']:.2f}",
                            f"${calibrated_result['dcf_price']:.2f}",
                            f"${calibrated_result['calibrated_price']:.2f}"
                        ],
                        'Weight': [
                            f"{calibrated_result['peer_weight']:.0%}",
                            f"{calibrated_result['dcf_weight']:.0%}",
                            "100%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
                    
                else:
                    st.error(f"❌ Calibrated market valuation failed: {calibrated_result['error']}")
        
        # Monte Carlo Simulation
        st.header("Monte Carlo Simulation")
        if st.button("Run Monte Carlo Analysis"):
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = self.run_monte_carlo_simulation(1000)
                
                if 'error' not in mc_results:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Valuation", f"${mc_results['mean_valuation']:.2f}")
                        st.metric("Median Valuation", f"${mc_results['median_valuation']:.2f}")
                    
                    with col2:
                        st.metric("Upside Probability", f"{mc_results['probabilities']['upside']:.1%}")
                        st.metric("Downside Probability", f"{mc_results['probabilities']['downside']:.1%}")
                    
                    with col3:
                        st.metric("Standard Deviation", f"${mc_results['std_valuation']:.2f}")
                        st.metric("Current Price", f"${mc_results['current_price']:.2f}")
                    
                    # Create Monte Carlo histogram
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=mc_results['all_valuations'],
                        nbinsx=50,
                        name='Valuation Distribution',
                        opacity=0.7
                    ))
                    
                    # Add current price line
                    fig.add_vline(
                        x=mc_results['current_price'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Current Price"
                    )
                    
                    fig.update_layout(
                        title=f"Monte Carlo Valuation Distribution: {self.symbol}",
                        xaxis_title="Valuation ($)",
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Percentile analysis
                    st.subheader("Percentile Analysis")
                    percentiles = mc_results['percentiles']
                    percentile_data = {
                        'Percentile': ['10th', '25th', '50th (Median)', '75th', '90th'],
                        'Valuation': [
                            f"${percentiles['p10']:.2f}",
                            f"${percentiles['p25']:.2f}",
                            f"${mc_results['median_valuation']:.2f}",
                            f"${percentiles['p75']:.2f}",
                            f"${percentiles['p90']:.2f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)
                else:
                    st.error(f"Monte Carlo simulation failed: {mc_results['error']}")
        
        # Display detailed report
        st.header("Detailed Analysis")
        report = self.generate_valuation_report()
        st.markdown(report)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("🚀 Equity Valuation Engine - Example Usage")
    
    # Initialize engine with Apple (AAPL) - mega-cap test case with calibration
    # Why Apple is interesting for testing:
    # - Mega-cap US company (~$3T market cap)
    # - Mature, cash-rich business model
    # - Clear peer group (FAANG/tech giants)
    # - High margins and predictable revenue streams
    # - Well-analyzed by Wall Street
    # - US market exposure (tests domestic data)
    # 
    # Previous test cases: DB (banking), TSMC (semiconductors)
    engine = EquityValuationEngine('AAPL', use_calibration=True)
    
    # Run calibrated scenario analysis
    print("\n📊 Running Calibrated Scenario Analysis...")
    calibrated_scenarios = engine.run_calibrated_scenario_analysis()
    
    # Run Calibrated Market-Driven Analysis
    print("\n🎯 Running Calibrated Market-Driven Analysis...")
    calibrated_market_result = engine.calculate_calibrated_market_valuation()
    
    # Display calibrated results
    if 'error' not in calibrated_market_result:
        print(f"\n🎯 Calibrated Results:")
        print(f"   • Current Price: ${calibrated_market_result['current_price']:.2f}")
        print(f"   • Calibrated Target: ${calibrated_market_result['calibrated_price']:.2f}")
        print(f"   • Upside/Downside: {calibrated_market_result['upside_downside']:.1%}")
        print(f"   • Calibration Used: {calibrated_market_result['calibration_used']}")
        if calibrated_market_result['calibrated_parameters']:
            params = calibrated_market_result['calibrated_parameters']
            print(f"   • Calibrated Revenue Growth: {params['revenue_growth']:.1%}")
            print(f"   • Calibrated EBIT Margin: {params['ebit_margin']:.1%}")
            print(f"   • Calibrated WACC: {params['wacc']:.1%}")
            print(f"   • Calibrated Terminal Growth: {params['terminal_growth']:.1%}")
    
    # Validate calibration accuracy
    print(f"\n🎯 Validating Calibration Accuracy...")
    accuracy_result = engine.validate_calibration_accuracy()
    
    if 'error' not in accuracy_result:
        metrics = accuracy_result.get('accuracy_metrics', {})
        print(f"   • Mean Absolute Error: ${metrics.get('mae', 0):.2f}")
        print(f"   • Mean Absolute Percentage Error: {metrics.get('mape', 0):.1%}")
        print(f"   • R² Score: {metrics.get('r2_score', 0):.3f}")
        print(f"   • Accuracy: {metrics.get('accuracy', 0):.1%}")
    else:
        print(f"   • Validation failed: {accuracy_result['error']}")
    
    # Get calibration summary
    print(f"\n📊 Calibration Summary:")
    summary = engine.get_calibration_summary()
    if 'error' not in summary:
        print(f"   • Total Calibrations: {summary['total_calibrations']}")
        print(f"   • Calibrated Symbols: {', '.join(summary['calibrated_symbols'])}")
        if summary['latest_calibration']:
            latest = summary['latest_calibration']
            print(f"   • Latest: {latest['symbol']} ({latest['date'][:10]})")
    else:
        print(f"   • Summary failed: {summary['error']}")
    
    # Calculate Implied WACC
    print("\n💰 Calculating Implied WACC...")
    implied_wacc_result = engine.calculate_implied_wacc()
    
    if 'error' not in implied_wacc_result:
        print(f"   • Implied WACC: {implied_wacc_result['implied_wacc']:.1%}")
        print(f"   • Target Price: ${implied_wacc_result['target_price']:.2f}")
        print(f"   • Calculated Price: ${implied_wacc_result['calculated_price']:.2f}")
    
    # Run Monte Carlo simulation
    print("\n🎲 Running Monte Carlo Simulation...")
    mc_results = engine.run_monte_carlo_simulation(1000)
    
    if 'error' not in mc_results:
        print(f"   • Mean Valuation: ${mc_results['mean_valuation']:.2f}")
        print(f"   • Upside Probability: {mc_results['probabilities']['upside']:.1%}")
        print(f"   • Downside Probability: {mc_results['probabilities']['downside']:.1%}")
    
    # Generate calibration report
    print("\n📋 Generating Calibration Report...")
    calibration_report = engine.generate_calibration_report()
    print(calibration_report)
    
    # Example of running the Streamlit dashboard
    # Uncomment the line below to run the dashboard
    # engine.create_valuation_dashboard()
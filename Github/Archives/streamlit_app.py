#!/usr/bin/env python3
"""
Equity Valuation Platform - Streamlit Dashboard
Professional equity valuation tool with DCF modeling, peer analysis, and calibration

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
from datetime import datetime
import sys
import os

# Add the Files directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Files'))

from equity_valuation import EquityValuationEngine
from valuation_calibration import ValuationCalibrator

# Page configuration
st.set_page_config(
    page_title="Equity Valuation Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #2563eb;
    }
    
    .success-metric {
        border-left-color: #10b981;
    }
    
    .warning-metric {
        border-left-color: #f59e0b;
    }
    
    .danger-metric {
        border-left-color: #ef4444;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1f2937, #374151);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = None
if 'calibration_results' not in st.session_state:
    st.session_state.calibration_results = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Equity Valuation Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for symbol input and parameters
    with st.sidebar:
        st.header("🔧 Analysis Configuration")
        
        # Symbol input
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            placeholder="Enter ticker symbol (e.g., AAPL, MSFT, TSLA)",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        if symbol and symbol != st.session_state.symbol:
            st.session_state.symbol = symbol
            st.session_state.engine = None  # Reset engine for new symbol
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Market-Driven Analysis", "Calibrated Analysis", "Scenario Analysis", "Monte Carlo"]
        )
        
        # Parameter configuration
        st.subheader("📈 Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            revenue_growth = st.slider("Revenue Growth (%)", 0.0, 25.0, 8.0) / 100
            ebit_margin = st.slider("EBIT Margin (%)", 5.0, 40.0, 15.0) / 100
        
        with col2:
            wacc = st.slider("WACC (%)", 3.0, 20.0, 10.0) / 100
            terminal_growth = st.slider("Terminal Growth (%)", 1.0, 8.0, 3.5) / 100
        
        # Custom multiples option
        use_custom_multiples = st.checkbox("Use Custom Multiples", False)
        if use_custom_multiples:
            st.subheader("🎯 Custom Multiples")
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_pe = st.number_input("P/E Ratio", min_value=5.0, max_value=50.0, value=20.0, step=0.5)
            with col2:
                custom_ev_rev = st.number_input("EV/Revenue", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
            with col3:
                custom_ev_ebitda = st.number_input("EV/EBITDA", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
        else:
            custom_pe = custom_ev_rev = custom_ev_ebitda = None
        
        # Run analysis button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if run_analysis and symbol:
        with st.spinner(f"Running {analysis_type} for {symbol}..."):
            try:
                # Initialize engine if needed
                if st.session_state.engine is None or st.session_state.symbol != symbol:
                    st.session_state.engine = EquityValuationEngine(symbol, use_calibration=True)
                
                engine = st.session_state.engine
                
                # Run analysis based on type
                if analysis_type == "Market-Driven Analysis":
                    run_market_driven_analysis(engine, revenue_growth, ebit_margin, wacc, terminal_growth, 
                                             custom_pe, custom_ev_rev, custom_ev_ebitda)
                elif analysis_type == "Calibrated Analysis":
                    run_calibrated_analysis(engine)
                elif analysis_type == "Scenario Analysis":
                    run_scenario_analysis(engine)
                elif analysis_type == "Monte Carlo":
                    run_monte_carlo_analysis(engine)
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Try a different symbol or check your internet connection")
    
    elif symbol and not run_analysis:
        # Show placeholder
        st.info(f"👆 Configure parameters in the sidebar and click 'Run Analysis' to analyze {symbol}")
        
        # Show company info if available
        if st.session_state.engine:
            show_company_overview(st.session_state.engine)

def run_market_driven_analysis(engine, revenue_growth, ebit_margin, wacc, terminal_growth, custom_pe, custom_ev_rev, custom_ev_ebitda):
    """Run market-driven valuation analysis"""
    
    # Prepare custom multiples
    custom_multiples = None
    if custom_pe or custom_ev_rev or custom_ev_ebitda:
        custom_multiples = {}
        if custom_pe: custom_multiples['pe_ratio'] = custom_pe
        if custom_ev_rev: custom_multiples['ev_revenue'] = custom_ev_rev
        if custom_ev_ebitda: custom_multiples['ev_ebitda'] = custom_ev_ebitda
    
    # Run market-aligned valuation
    result = engine.calculate_market_aligned_valuation(
        custom_multiples=custom_multiples,
        target_wacc=wacc
    )
    
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Display results
    st.header("🎯 Market-Driven Valuation Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${result['current_price']:.2f}")
    
    with col2:
        st.metric("Target Price", f"${result['blended_price']:.2f}")
    
    with col3:
        upside_downside = result['upside_downside']
        st.metric(
            "Upside/Downside", 
            f"{upside_downside:.1%}",
            delta=f"{upside_downside:.1%}" if upside_downside != 0 else None
        )
    
    with col4:
        st.metric("WACC Used", f"{result['target_wacc']:.1%}")
    
    # Detailed breakdown
    st.subheader("📊 Valuation Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Method comparison
        methods_data = {
            'Method': ['Peer Multiples', 'DCF', 'Blended'],
            'Price': [f"${result['peer_price']:.2f}", f"${result['dcf_price']:.2f}", f"${result['blended_price']:.2f}"],
            'Weight': [f"{result['peer_weight']:.0%}", f"{result['dcf_weight']:.0%}", "100%"]
        }
        st.dataframe(pd.DataFrame(methods_data), use_container_width=True)
    
    with col2:
        # Price comparison chart
        fig = go.Figure()
        fig.add_bar(
            x=['Current', 'Peer', 'DCF', 'Target'],
            y=[result['current_price'], result['peer_price'], result['dcf_price'], result['blended_price']],
            marker_color=['#6b7280', '#3b82f6', '#10b981', '#2563eb']
        )
        fig.update_layout(
            title="Price Comparison",
            yaxis_title="Price ($)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def run_calibrated_analysis(engine):
    """Run calibrated valuation analysis"""
    
    st.header("🎯 Calibrated Valuation Analysis")
    
    # Run calibrated market valuation
    result = engine.calculate_calibrated_market_valuation()
    
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${result['current_price']:.2f}")
    
    with col2:
        st.metric("Calibrated Target", f"${result['calibrated_price']:.2f}")
    
    with col3:
        upside_downside = result['upside_downside']
        st.metric(
            "Upside/Downside", 
            f"{upside_downside:.1%}",
            delta=f"{upside_downside:.1%}" if upside_downside != 0 else None
        )
    
    with col4:
        st.metric("Calibration Used", "✅" if result['calibration_used'] else "❌")
    
    # Show calibrated parameters if available
    if result['calibrated_parameters']:
        st.subheader("📊 Calibrated Parameters")
        params = result['calibrated_parameters']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Revenue Growth", f"{params['revenue_growth']:.1%}")
        with col2:
            st.metric("EBIT Margin", f"{params['ebit_margin']:.1%}")
        with col3:
            st.metric("WACC", f"{params['wacc']:.1%}")
        with col4:
            st.metric("Terminal Growth", f"{params['terminal_growth']:.1%}")

def run_scenario_analysis(engine):
    """Run scenario analysis"""
    
    st.header("📈 Scenario Analysis")
    
    # Run scenario analysis
    results = engine.run_scenario_analysis()
    
    if not results:
        st.error("❌ Scenario analysis failed")
        return
    
    # Display scenario results
    for scenario_name, scenario_data in results.items():
        if 'error' in scenario_data:
            continue
            
        st.subheader(f"{scenario_name} Case")
        
        dcf = scenario_data.get('dcf', {})
        peer = scenario_data.get('peer', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("DCF Price", f"${dcf.get('dcf_price', 0):.2f}")
            st.metric("DCF Upside/Downside", f"{dcf.get('upside_downside', 0):.1%}")
        
        with col2:
            st.metric("Peer Price", f"${peer.get('peer_price', 0):.2f}")
            st.metric("Peer Upside/Downside", f"{peer.get('upside_downside', 0):.1%}")

def run_monte_carlo_analysis(engine):
    """Run Monte Carlo simulation"""
    
    st.header("🎲 Monte Carlo Simulation")
    
    # Run Monte Carlo simulation
    result = engine.run_monte_carlo_simulation(1000)
    
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Valuation", f"${result['mean_valuation']:.2f}")
        st.metric("Median Valuation", f"${result['median_valuation']:.2f}")
    
    with col2:
        st.metric("Upside Probability", f"{result['probabilities']['upside']:.1%}")
        st.metric("Downside Probability", f"{result['probabilities']['downside']:.1%}")
    
    with col3:
        st.metric("Standard Deviation", f"${result['std_valuation']:.2f}")
        st.metric("Current Price", f"${result['current_price']:.2f}")
    
    # Monte Carlo distribution chart
    fig = go.Figure()
    fig.add_histogram(
        x=result['all_valuations'],
        nbinsx=50,
        name='Valuation Distribution',
        opacity=0.7,
        marker_color='#3b82f6'
    )
    
    # Add current price line
    fig.add_vline(
        x=result['current_price'],
        line_dash="dash",
        line_color="red",
        annotation_text="Current Price"
    )
    
    fig.update_layout(
        title=f"Monte Carlo Valuation Distribution: {engine.symbol}",
        xaxis_title="Valuation ($)",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_company_overview(engine):
    """Show company overview information"""
    
    try:
        current_data = engine._extract_current_financials()
        
        if current_data:
            st.subheader("📋 Company Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Market Cap", f"${current_data.get('market_cap', 0)/1_000_000_000:.1f}B")
            
            with col2:
                st.metric("Revenue", f"${current_data.get('revenue', 0)/1_000_000_000:.1f}B")
            
            with col3:
                st.metric("EPS", f"${current_data.get('eps', 0):.2f}")
            
            with col4:
                pe_ratio = current_data.get('current_price', 0) / current_data.get('eps', 1) if current_data.get('eps', 0) > 0 else 0
                st.metric("P/E Ratio", f"{pe_ratio:.1f}x")
    
    except Exception as e:
        st.warning(f"Could not load company overview: {e}")

if __name__ == "__main__":
    main()

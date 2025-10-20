"""
Streamlit Web Interface for Equity Derivatives Pricing Project
Interactive dashboard for options strategy analysis and risk management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our main project modules
from EQD_Project import *

# Page configuration
st.set_page_config(
    page_title="Equity Derivatives Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .strategy-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Equity Derivatives Pricing Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Symbol selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    expiration_days = st.sidebar.slider("Days to Expiration", 7, 365, 30)
    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        ["ATM Straddle", "Bull Call Spread", "Bear Put Spread", "Strangle", "Synthetic Long"]
    )
    
    # Risk parameters
    st.sidebar.subheader("Risk Parameters")
    max_loss = st.sidebar.number_input("Max Loss ($)", value=500, step=100)
    position_size = st.sidebar.number_input("Position Size", value=1, min_value=1, max_value=10)
    
    # Initialize system
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
        st.session_state.volatility_model = VolatilityModel(st.session_state.data_loader)
        st.session_state.strategy_constructor = StrategyConstructor(st.session_state.data_loader, st.session_state.volatility_model)
        st.session_state.greeks_engine = GreeksEngine(st.session_state.data_loader)
        st.session_state.pnl_simulator = PnLSimulator(st.session_state.data_loader, st.session_state.greeks_engine)
        st.session_state.performance_analyzer = PerformanceAnalyzer(st.session_state.data_loader)
    
    # Load data
    with st.spinner("Loading market data..."):
        try:
            data = st.session_state.data_loader.load_stock_data(symbol, start_date, end_date)
            options_data = st.session_state.data_loader.load_options_chain(symbol)
            
            if data is not None and options_data is not None:
                st.success(f"✅ Data loaded successfully for {symbol}")
            else:
                st.error(f"❌ Failed to load data for {symbol}")
                return
                
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", 
        "🎯 Strategy Builder", 
        "⚡ Risk Analysis", 
        "💰 PnL Simulation", 
        "📈 Performance"
    ])
    
    with tab1:
        show_market_overview(symbol, data, options_data)
    
    with tab2:
        show_strategy_builder(symbol, strategy_type, expiration_days, position_size)
    
    with tab3:
        show_risk_analysis(symbol, expiration_days)
    
    with tab4:
        show_pnl_simulation(symbol, strategy_type, expiration_days, max_loss)
    
    with tab5:
        show_performance_analysis()


def show_market_overview(symbol, data, options_data):
    """Display market overview tab"""
    st.header("📊 Market Overview")
    
    # Current market data
    current_price = data['Close'].iloc[-1]
    current_vol = data['Volatility'].iloc[-1]
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Current Volatility", f"{current_vol:.1%}")
    
    with col3:
        price_change = data['Close'].pct_change().iloc[-1] * 100
        st.metric("Daily Change", f"{price_change:+.2f}%")
    
    with col4:
        if len(options_data) > 0:
            avg_iv = options_data['IV'].mean()
            st.metric("Avg IV", f"{avg_iv:.1%}")
    
    # Price and volatility chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stock Price', 'Volatility'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Volatility'], name='Volatility', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Options chain overview
    if len(options_data) > 0:
        st.subheader("Options Chain Overview")
        
        # Filter by expiration
        expirations = options_data['Expiration'].unique()
        selected_exp = st.selectbox("Select Expiration", expirations)
        
        exp_data = options_data[options_data['Expiration'] == selected_exp]
        
        # IV surface
        fig = px.scatter(
            exp_data, 
            x='Strike', 
            y='IV', 
            color='Type',
            title=f'IV Surface - {selected_exp} DTE'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_strategy_builder(symbol, strategy_type, expiration_days, position_size):
    """Display strategy builder tab"""
    st.header("🎯 Strategy Builder")
    
    current_price = st.session_state.data_loader.price_data[symbol]['Close'].iloc[-1]
    
    # Strategy creation
    strategy = None
    
    if strategy_type == "ATM Straddle":
        strategy = st.session_state.strategy_constructor.create_atm_straddle(
            symbol, expiration_days, position_size
        )
    elif strategy_type == "Bull Call Spread":
        col1, col2 = st.columns(2)
        with col1:
            short_strike = st.number_input("Short Strike", value=current_price, step=1.0)
        with col2:
            long_strike = st.number_input("Long Strike", value=current_price + 10, step=1.0)
        
        strategy = st.session_state.strategy_constructor.create_vertical_spread(
            symbol, "Call", short_strike, long_strike, expiration_days, position_size
        )
    elif strategy_type == "Bear Put Spread":
        col1, col2 = st.columns(2)
        with col1:
            short_strike = st.number_input("Short Strike", value=current_price, step=1.0)
        with col2:
            long_strike = st.number_input("Long Strike", value=current_price - 10, step=1.0)
        
        strategy = st.session_state.strategy_constructor.create_vertical_spread(
            symbol, "Put", short_strike, long_strike, expiration_days, position_size
        )
    elif strategy_type == "Strangle":
        col1, col2 = st.columns(2)
        with col1:
            call_strike = st.number_input("Call Strike", value=current_price + 10, step=1.0)
        with col2:
            put_strike = st.number_input("Put Strike", value=current_price - 10, step=1.0)
        
        strategy = st.session_state.strategy_constructor.create_strangle(
            symbol, call_strike, put_strike, expiration_days, position_size
        )
    elif strategy_type == "Synthetic Long":
        strike = st.number_input("Strike Price", value=current_price, step=1.0)
        strategy = st.session_state.strategy_constructor.create_synthetic_long(
            symbol, strike, expiration_days, position_size
        )
    
    if strategy:
        st.success(f"✅ {strategy['type']} created successfully!")
        
        # Strategy details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cost", f"${strategy['total_cost']:.2f}")
        
        with col2:
            st.metric("Days to Expiry", strategy['expiration_days'])
        
        with col3:
            st.metric("Position Size", position_size)
        
        # Strategy legs
        st.subheader("Strategy Legs")
        legs_df = pd.DataFrame(strategy['legs'])
        st.dataframe(legs_df, use_container_width=True)
        
        # Store strategy in session state
        st.session_state.current_strategy = strategy
    else:
        st.error("❌ Failed to create strategy")


def show_risk_analysis(symbol, expiration_days):
    """Display risk analysis tab"""
    st.header("⚡ Risk Analysis")
    
    if 'current_strategy' not in st.session_state:
        st.warning("⚠️ Please create a strategy first in the Strategy Builder tab")
        return
    
    strategy = st.session_state.current_strategy
    current_price = st.session_state.data_loader.price_data[symbol]['Close'].iloc[-1]
    
    # Greeks calculation
    greeks = st.session_state.greeks_engine.calculate_strategy_greeks(
        strategy, current_price, expiration_days
    )
    
    # Greeks metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Delta", f"{greeks['delta']:.3f}")
    
    with col2:
        st.metric("Gamma", f"{greeks['gamma']:.3f}")
    
    with col3:
        st.metric("Theta", f"{greeks['theta']:.3f}")
    
    with col4:
        st.metric("Vega", f"{greeks['vega']:.3f}")
    
    # Greeks evolution chart
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    greeks_df = st.session_state.greeks_engine.track_greeks_evolution(
        strategy, price_range, expiration_days
    )
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=greeks_df['price'], y=greeks_df['delta'], name='Delta'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=greeks_df['price'], y=greeks_df['gamma'], name='Gamma'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=greeks_df['price'], y=greeks_df['theta'], name='Theta'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=greeks_df['price'], y=greeks_df['vega'], name='Vega'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_pnl_simulation(symbol, strategy_type, expiration_days, max_loss):
    """Display PnL simulation tab"""
    st.header("💰 PnL Simulation")
    
    if 'current_strategy' not in st.session_state:
        st.warning("⚠️ Please create a strategy first in the Strategy Builder tab")
        return
    
    strategy = st.session_state.current_strategy
    current_price = st.session_state.data_loader.price_data[symbol]['Close'].iloc[-1]
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        price_range_start = st.number_input("Price Range Start", value=current_price * 0.7, step=1.0)
        price_range_end = st.number_input("Price Range End", value=current_price * 1.3, step=1.0)
    
    with col2:
        implied_move = st.number_input("Implied Move ($)", value=current_price * 0.05, step=0.5)
        num_points = st.slider("Number of Price Points", 50, 200, 100)
    
    price_range = np.linspace(price_range_start, price_range_end, num_points)
    
    # PnL simulation
    pnl_df = st.session_state.pnl_simulator.simulate_pnl_grid(strategy, price_range, implied_move)
    
    # Payoff diagram
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_df['price'],
        y=pnl_df['payoff'],
        mode='lines',
        name='Strategy Payoff',
        line=dict(color='blue', width=3)
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    # Implied move lines
    fig.add_vline(x=current_price + implied_move, line_dash="dash", line_color="red", 
                  annotation_text=f"+{implied_move:.1f}")
    fig.add_vline(x=current_price - implied_move, line_dash="dash", line_color="red",
                  annotation_text=f"-{implied_move:.1f}")
    
    fig.update_layout(
        title=f"{strategy['type']} Payoff Diagram",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics
    risk_metrics = st.session_state.pnl_simulator.calculate_risk_metrics(
        strategy, price_range, implied_move
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Profit", f"${risk_metrics['max_profit']:.2f}")
    
    with col2:
        st.metric("Max Loss", f"${risk_metrics['max_loss']:.2f}")
    
    with col3:
        st.metric("Expected Return", f"${risk_metrics['expected_return']:.2f}")
    
    with col4:
        st.metric("Probability of Profit", f"{risk_metrics['prob_profit']:.1%}")
    
    # Breakeven points
    if risk_metrics['breakeven_points']:
        st.subheader("Breakeven Points")
        for i, be_point in enumerate(risk_metrics['breakeven_points']):
            st.write(f"Breakeven {i+1}: ${be_point:.2f}")


def show_performance_analysis():
    """Display performance analysis tab"""
    st.header("📈 Performance Analysis")
    
    # Add sample trades for demonstration
    if 'trades_added' not in st.session_state:
        sample_trades = [
            {'strategy_type': 'Straddle', 'pnl': 150, 'initial_cost': 500},
            {'strategy_type': 'Call Spread', 'pnl': -50, 'initial_cost': 200},
            {'strategy_type': 'Strangle', 'pnl': 300, 'initial_cost': 400},
            {'strategy_type': 'Straddle', 'pnl': -100, 'initial_cost': 600},
            {'strategy_type': 'Put Spread', 'pnl': 75, 'initial_cost': 150},
            {'strategy_type': 'Strangle', 'pnl': 200, 'initial_cost': 350},
            {'strategy_type': 'Call Spread', 'pnl': 125, 'initial_cost': 250},
            {'strategy_type': 'Straddle', 'pnl': -75, 'initial_cost': 450}
        ]
        
        for trade in sample_trades:
            st.session_state.performance_analyzer.add_trade(trade)
        
        st.session_state.trades_added = True
    
    # Performance metrics
    performance = st.session_state.performance_analyzer.calculate_trade_performance()
    
    if performance:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", performance['total_trades'])
            st.metric("Win Rate", f"{performance['win_rate']:.1%}")
        
        with col2:
            st.metric("Total PnL", f"${performance['total_pnl']:.2f}")
            st.metric("Avg Win", f"${performance['avg_win']:.2f}")
        
        with col3:
            st.metric("Avg Loss", f"${performance['avg_loss']:.2f}")
            st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"${performance['max_drawdown']:.2f}")
        
        # Performance charts
        trades_df = pd.DataFrame(st.session_state.performance_analyzer.trades_history)
        
        # Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(cumulative_pnl))),
            y=cumulative_pnl,
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="Cumulative PnL Over Time",
            xaxis_title="Trade Number",
            yaxis_title="Cumulative PnL ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # PnL distribution
        fig = px.histogram(
            trades_df, 
            x='pnl', 
            nbins=20,
            title="PnL Distribution",
            labels={'pnl': 'PnL ($)', 'count': 'Frequency'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Win rate by strategy
        strategy_performance = trades_df.groupby('strategy_type').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).mean()]
        }).round(3)
        
        strategy_performance.columns = ['Total Trades', 'Total PnL', 'Win Rate']
        st.subheader("Performance by Strategy")
        st.dataframe(strategy_performance, use_container_width=True)


if __name__ == "__main__":
    main() 
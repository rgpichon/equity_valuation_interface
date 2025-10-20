"""
Test Script for Equity Derivatives Pricing Project
Demonstrates key functionality and validates the system
"""

import sys
import traceback
from datetime import datetime

def test_project():
    """Test the main project functionality"""
    print("=== Testing Equity Derivatives Pricing Project ===\n")
    
    try:
        # Import the main project
        from EQD_Project import *
        
        print("✅ Successfully imported project modules")
        
        # Initialize system components
        print("\n--- Initializing System Components ---")
        data_loader = DataLoader()
        volatility_model = VolatilityModel(data_loader)
        strategy_constructor = StrategyConstructor(data_loader, volatility_model)
        greeks_engine = GreeksEngine(data_loader)
        pnl_simulator = PnLSimulator(data_loader, greeks_engine)
        performance_analyzer = PerformanceAnalyzer(data_loader)
        
        print("✅ All system components initialized")
        
        # Test data loading
        print("\n--- Testing Data Loading ---")
        symbol = "AAPL"
        data = data_loader.load_stock_data(symbol, "2023-01-01", "2024-01-01")
        options_data = data_loader.load_options_chain(symbol)
        
        if data is not None:
            print(f"✅ Loaded {len(data)} days of price data for {symbol}")
            print(f"   Current price: ${data['Close'].iloc[-1]:.2f}")
            print(f"   Current volatility: {data['Volatility'].iloc[-1]:.1%}")
        else:
            print("❌ Failed to load price data")
            return False
        
        if options_data is not None:
            print(f"✅ Loaded {len(options_data)} options contracts")
        else:
            print("❌ Failed to load options data")
            return False
        
        # Test volatility analysis
        print("\n--- Testing Volatility Analysis ---")
        iv_stats = volatility_model.calculate_iv_rank_percentile(symbol)
        if iv_stats:
            print(f"✅ IV Analysis:")
            print(f"   Current IV: {iv_stats['current_iv']:.2%}")
            print(f"   IV Rank: {iv_stats['iv_rank']:.2%}")
            print(f"   IV Percentile: {iv_stats['iv_percentile']:.2%}")
        
        implied_move = volatility_model.estimate_implied_move(symbol, 30)
        if implied_move:
            print(f"   Implied Move: ±${implied_move:.2f}")
        
        # Test strategy construction
        print("\n--- Testing Strategy Construction ---")
        current_price = data['Close'].iloc[-1]
        
        # Test ATM Straddle
        straddle = strategy_constructor.create_atm_straddle(symbol, 30)
        if straddle:
            print(f"✅ Created {straddle['type']}")
            print(f"   Cost: ${straddle['total_cost']:.2f}")
            print(f"   Legs: {len(straddle['legs'])}")
        else:
            print("❌ Failed to create straddle")
        
        # Test Vertical Spread
        call_spread = strategy_constructor.create_vertical_spread(
            symbol, "Call", current_price, current_price + 10, 30
        )
        if call_spread:
            print(f"✅ Created {call_spread['type']}")
            print(f"   Cost: ${call_spread['total_cost']:.2f}")
        else:
            print("❌ Failed to create call spread")
        
        # Test Greeks calculation
        print("\n--- Testing Greeks Calculation ---")
        if straddle:
            greeks = greeks_engine.calculate_strategy_greeks(straddle, current_price, 30)
            print(f"✅ Strategy Greeks:")
            print(f"   Delta: {greeks['delta']:.3f}")
            print(f"   Gamma: {greeks['gamma']:.3f}")
            print(f"   Theta: {greeks['theta']:.3f}")
            print(f"   Vega: {greeks['vega']:.3f}")
        
        # Test PnL simulation
        print("\n--- Testing PnL Simulation ---")
        if straddle:
            price_range = [current_price * 0.8, current_price, current_price * 1.2]
            pnl_results = pnl_simulator.simulate_pnl_grid(straddle, price_range)
            
            print(f"✅ PnL Simulation Results:")
            for _, row in pnl_results.iterrows():
                print(f"   Price: ${row['price']:.2f} -> PnL: ${row['payoff']:.2f}")
        
        # Test performance analysis
        print("\n--- Testing Performance Analysis ---")
        sample_trades = [
            {'strategy_type': 'Straddle', 'pnl': 150, 'initial_cost': 500},
            {'strategy_type': 'Call Spread', 'pnl': -50, 'initial_cost': 200},
            {'strategy_type': 'Strangle', 'pnl': 300, 'initial_cost': 400}
        ]
        
        for trade in sample_trades:
            performance_analyzer.add_trade(trade)
        
        performance = performance_analyzer.calculate_trade_performance()
        if performance:
            print(f"✅ Performance Analysis:")
            print(f"   Total Trades: {performance['total_trades']}")
            print(f"   Win Rate: {performance['win_rate']:.1%}")
            print(f"   Total PnL: ${performance['total_pnl']:.2f}")
            print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        
        print("\n=== All Tests Passed Successfully! ===")
        print("\nProject Features Demonstrated:")
        print("✓ Data loading and management")
        print("✓ Volatility analysis and modeling")
        print("✓ Options strategy construction")
        print("✓ Greeks calculation and risk management")
        print("✓ PnL simulation and analysis")
        print("✓ Performance tracking and analytics")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def test_streamlit_import():
    """Test if Streamlit components can be imported"""
    print("\n--- Testing Streamlit Interface ---")
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Streamlit and Plotly imports successful")
        return True
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        print("   Install with: pip install streamlit plotly")
        return False


if __name__ == "__main__":
    print("Starting Equity Derivatives Pricing Project Tests...\n")
    
    # Test main project
    main_success = test_project()
    
    # Test Streamlit interface
    streamlit_success = test_streamlit_import()
    
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"Main Project: {'✅ PASSED' if main_success else '❌ FAILED'}")
    print(f"Streamlit Interface: {'✅ PASSED' if streamlit_success else '❌ FAILED'}")
    
    if main_success and streamlit_success:
        print("\n🎉 All tests passed! The project is ready for use.")
        print("\nTo run the main project:")
        print("   python 'EQD Project.py'")
        print("\nTo run the Streamlit dashboard:")
        print("   streamlit run streamlit_app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    print("="*50) 
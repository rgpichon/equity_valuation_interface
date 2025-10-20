# Equity Derivatives Pricing Project

A comprehensive Python system for options pricing, strategy construction, and risk management designed for quantitative finance applications.

## 🎯 Project Overview

This project implements a modular equity derivatives pricing system that demonstrates advanced financial engineering concepts including:

- **Data Management**: Real-time market data ingestion and historical analysis
- **Volatility Modeling**: Implied volatility analysis and event-driven volatility forecasting
- **Strategy Construction**: Dynamic options strategy builder with risk parameterization
- **Risk Management**: Greeks calculation and real-time risk monitoring
- **PnL Simulation**: Monte Carlo and scenario-based profit/loss analysis
- **Performance Analytics**: Comprehensive trade performance tracking and visualization

## 🏗️ Architecture

The system is structured into 7 modular components:

### 1. Data Loader (`DataLoader`)
- **Real-time data ingestion** from Yahoo Finance API
- **Historical price data** with volatility calculations
- **Options chain management** with synthetic data generation
- **Event calendar integration** for earnings and macro events
- **CSV import/export** capabilities

### 2. Volatility Model (`VolatilityModel`)
- **ATM straddle IV calculation** for implied move estimation
- **Historical volatility analysis** with rolling windows
- **IV rank and percentile** calculations
- **Event-driven volatility** pre/post analysis
- **Volatility visualization** with comprehensive charts

### 3. Strategy Constructor (`StrategyConstructor`)
- **ATM Straddles & Strangles** with dynamic pricing
- **Vertical Call/Put Spreads** with risk management
- **Synthetic Longs/Shorts** for directional exposure
- **Calendar spreads** for time decay strategies
- **Parameterized risk limits** and position sizing

### 4. Greeks Engine (`GreeksEngine`)
- **Black-Scholes Greeks** calculation (Delta, Gamma, Theta, Vega)
- **Portfolio-level Greeks** aggregation
- **Real-time Greeks evolution** tracking
- **Risk sensitivity analysis** across price ranges
- **Greeks visualization** with interactive charts

### 5. PnL Simulator (`PnLSimulator`)
- **Monte Carlo simulation** for strategy outcomes
- **Scenario analysis** with implied move integration
- **Transaction cost modeling** and slippage analysis
- **Breakeven point calculation** and risk metrics
- **Payoff diagram generation** with profit zones

### 6. Performance Analyzer (`PerformanceAnalyzer`)
- **Trade-by-trade performance** tracking
- **Win/loss ratio analysis** by strategy type
- **Rolling Sharpe ratio** and drawdown calculations
- **Volatility crush analysis** post-events
- **Performance heatmaps** by event size and IV level

### 7. Streamlit Interface (Optional)
- **Interactive web dashboard** for strategy selection
- **Real-time PnL monitoring** and risk alerts
- **Custom risk preferences** and position sizing
- **Dynamic visualization** of Greeks and payoffs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd equity-derivatives-pricing

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Import the main project
from EQD_Project import *

# Initialize the system
data_loader = DataLoader()
volatility_model = VolatilityModel(data_loader)
strategy_constructor = StrategyConstructor(data_loader, volatility_model)
greeks_engine = GreeksEngine(data_loader)
pnl_simulator = PnLSimulator(data_loader, greeks_engine)
performance_analyzer = PerformanceAnalyzer(data_loader)

# Load market data
symbol = "AAPL"
data_loader.load_stock_data(symbol, "2023-01-01", "2024-01-01")
data_loader.load_options_chain(symbol)

# Create and analyze a strategy
straddle = strategy_constructor.create_atm_straddle(symbol, 30)
if straddle:
    # Calculate Greeks
    greeks = greeks_engine.calculate_strategy_greeks(straddle, 150, 30)
    
    # Simulate PnL
    price_range = np.linspace(140, 160, 100)
    pnl_simulator.plot_payoff_diagram(straddle, price_range)
```

### Advanced Usage

```python
# Volatility analysis
iv_stats = volatility_model.calculate_iv_rank_percentile(symbol)
implied_move = volatility_model.estimate_implied_move(symbol, 30)

# Create complex strategies
call_spread = strategy_constructor.create_vertical_spread(
    symbol, "Call", 150, 160, 30
)

# Track performance
performance_analyzer.add_trade({
    'strategy_type': 'Straddle',
    'pnl': 150,
    'initial_cost': 500
})
performance_analyzer.plot_performance_summary()
```

## 📊 Key Features

### Data Management
- **Real-time market data** from multiple sources
- **Historical volatility** calculations with customizable windows
- **Options chain** management with synthetic data generation
- **Event calendar** integration for earnings and macro events

### Volatility Analysis
- **Implied volatility** calculation using ATM straddle pricing
- **IV rank and percentile** analysis for relative positioning
- **Event-driven volatility** analysis pre/post earnings
- **Volatility surface** visualization and analysis

### Strategy Construction
- **Modular strategy builder** with risk parameterization
- **Multiple strategy types**: straddles, spreads, strangles, synthetics
- **Dynamic pricing** based on current market conditions
- **Risk limit enforcement** and position sizing

### Risk Management
- **Real-time Greeks** calculation and monitoring
- **Portfolio-level risk** aggregation
- **Scenario analysis** across multiple price ranges
- **Risk visualization** with interactive charts

### PnL Simulation
- **Monte Carlo simulation** for strategy outcomes
- **Implied move integration** for event-driven analysis
- **Transaction cost modeling** including slippage
- **Breakeven analysis** and risk metrics calculation

### Performance Analytics
- **Comprehensive trade tracking** with detailed metrics
- **Strategy-specific performance** analysis
- **Risk-adjusted returns** calculation (Sharpe ratio, drawdown)
- **Visualization suite** for performance analysis

## 📈 Example Outputs

### Volatility Analysis
- Current IV: 25.3%
- IV Rank: 67.2%
- IV Percentile: 73.1%
- Implied Move: ±$7.50

### Strategy Metrics
- Max Profit: $450.00
- Max Loss: -$200.00
- Expected Return: $125.00
- Probability of Profit: 65.2%
- Breakeven Points: $142.50, $157.50

### Performance Summary
- Total Trades: 25
- Win Rate: 68.0%
- Total PnL: $2,450.00
- Sharpe Ratio: 1.85
- Max Drawdown: -$750.00

## 🛠️ Technical Implementation

### Core Algorithms
- **Black-Scholes pricing** with Greeks calculation
- **Implied volatility** estimation using Newton-Raphson
- **Monte Carlo simulation** for PnL analysis
- **Rolling statistics** for volatility analysis

### Data Structures
- **Pandas DataFrames** for time series data
- **NumPy arrays** for numerical computations
- **Custom classes** for strategy and trade objects
- **JSON configuration** for system parameters

### Visualization
- **Matplotlib/Seaborn** for static charts
- **Plotly** for interactive visualizations
- **Streamlit** for web dashboard (optional)
- **Custom plotting** functions for financial data

## 📚 Educational Value

This project demonstrates advanced concepts in:

- **Quantitative Finance**: Options pricing, Greeks, volatility modeling
- **Financial Engineering**: Strategy construction, risk management
- **Data Science**: Time series analysis, statistical modeling
- **Software Engineering**: Modular design, object-oriented programming
- **Risk Management**: Portfolio theory, scenario analysis

## 🎓 CV-Ready Features

### Technical Skills Demonstrated
- **Python Programming**: Advanced OOP, data structures, algorithms
- **Financial Modeling**: Options pricing, volatility analysis, Greeks
- **Data Analysis**: Pandas, NumPy, statistical analysis
- **Visualization**: Matplotlib, Seaborn, interactive charts
- **API Integration**: Real-time data feeds, external services

### Quantitative Finance Knowledge
- **Options Theory**: Black-Scholes, Greeks, implied volatility
- **Risk Management**: Portfolio theory, scenario analysis
- **Trading Strategies**: Event-driven, volatility-based strategies
- **Performance Analytics**: Sharpe ratio, drawdown, risk metrics

### Professional Development
- **Modular Architecture**: Scalable, maintainable code structure
- **Documentation**: Comprehensive README and code comments
- **Testing**: Robust error handling and validation
- **Deployment**: Requirements management and environment setup

## 🔧 Customization

### Adding New Strategies
```python
def create_custom_strategy(self, symbol, params):
    # Implement custom strategy logic
    strategy = {
        'type': 'Custom Strategy',
        'legs': [...],
        'total_cost': cost
    }
    return strategy
```

### Extending Data Sources
```python
def load_custom_data(self, source, symbol):
    # Implement custom data loading
    data = fetch_from_source(source, symbol)
    return self.process_data(data)
```

### Custom Risk Metrics
```python
def calculate_custom_risk(self, strategy):
    # Implement custom risk calculations
    risk_metrics = {...}
    return risk_metrics
```

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with relevant financial regulations when using for actual trading.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.

---

**Note**: This project is designed for educational purposes and demonstrates advanced quantitative finance concepts. It should not be used for actual trading without proper validation and risk management procedures. 
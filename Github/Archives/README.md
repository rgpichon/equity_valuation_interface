# 📊 Equity Valuation Platform

A professional-grade equity valuation platform combining DCF modeling, peer analysis, and parameter calibration with an intuitive Streamlit interface.

## 🚀 Features

- **DCF Valuation**: Discounted Cash Flow modeling with enhanced WACC calculation
- **Peer Analysis**: Market-based peer multiple analysis
- **Parameter Calibration**: Automated parameter optimization using historical data
- **Scenario Testing**: Base, Bull, and Bear case analysis
- **Monte Carlo Simulation**: Uncertainty modeling with probability distributions
- **Real-time Data**: Integration with Financial Modeling Prep API
- **Professional UI**: Clean, modern Streamlit dashboard

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd equity-valuation-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## 📈 Usage

1. **Enter Symbol**: Input a stock ticker symbol (e.g., AAPL, MSFT, TSLA)
2. **Configure Parameters**: Adjust revenue growth, margins, and WACC in the sidebar
3. **Select Analysis Type**:
   - **Market-Driven**: Combines peer multiples with DCF
   - **Calibrated**: Uses optimized parameters from historical data
   - **Scenario Analysis**: Compares multiple scenarios
   - **Monte Carlo**: Shows valuation uncertainty
4. **Run Analysis**: Click "Run Analysis" to generate results

## 🎯 Key Components

### Backend Engine (`Files/`)
- `equity_valuation.py`: Core valuation engine with DCF and peer analysis
- `valuation_calibration.py`: Parameter optimization and validation system

### Frontend (`streamlit_app.py`)
- Professional Streamlit dashboard
- Interactive parameter controls
- Real-time visualization with Plotly
- Responsive design for different screen sizes

## 📊 Example Outputs

- **Target Price**: Market-aligned valuation with confidence intervals
- **Upside/Downside**: Percentage deviation from current price
- **Peer Comparison**: Relative valuation against industry peers
- **Parameter Sensitivity**: Impact of key assumptions on valuation
- **Uncertainty Analysis**: Monte Carlo probability distributions

## 🔧 API Integration

The platform integrates with Financial Modeling Prep API for:
- Real-time stock prices and market data
- Financial statements (Income, Balance Sheet, Cash Flow)
- Peer company data and industry benchmarks
- Historical data for parameter calibration

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

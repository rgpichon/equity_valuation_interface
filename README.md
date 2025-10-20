# Stock Analysis Platform

A comprehensive stock analysis platform that combines DCF valuation, technical analysis, and analyst estimates into a unified dashboard interface.

## 🚀 Features

### Core Analysis Modules
- **DCF Valuation**: Complete discounted cash flow analysis with scenario modeling (Bear/Base/Bull)
- **Technical Analysis**: Advanced technical indicators including RSI, MACD, Bollinger Bands, ATR
- **Analyst Estimates**: Integration with analyst price targets and ratings
- **Risk Assessment**: Comprehensive risk metrics and signal strength analysis

### Dashboard Interface
- **Main Dashboard**: 2x2 quadrant layout for comprehensive overview
- **Market Overview**: Portfolio-level analysis and signal distribution
- **Individual Analysis**: Detailed single-stock deep dive
- **Comparison Tools**: Side-by-side stock comparison
- **Interactive Charts**: Real-time price charts with technical overlays

### Advanced Features
- **Scenario Analysis**: Bear/Base/Bull DCF projections with sensitivity analysis
- **Data Management**: Centralized data handling with caching
- **Export Capabilities**: CSV downloads for projections and sensitivity matrices
- **Real-time Updates**: Live data integration and analysis

## 📁 Project Structure

```
├── Files/
│   ├── interface.py                 # Main Streamlit dashboard interface
│   ├── analysis_combo.py            # Core analysis orchestration
│   ├── enhanced_comprehensive_data.py # Data management system
│   ├── dcf_calculation.py           # DCF valuation engine
│   ├── dcf_calibration.py           # Parameter calibration tools
│   ├── technical_analysis.py        # Technical indicators
│   ├── analyst_estimates_data.py    # Analyst data integration
│   └── requirements.txt             # Python dependencies
├── Archives/
│   ├── Files 2/                     # Legacy analysis modules
│   ├── streamlit_app.py            # Original Streamlit app
│   └── README.md                    # Archive documentation
└── tech_ana_chat.md                # Technical analysis documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd stock-analysis-platform
```

2. Install dependencies:
```bash
pip install -r Files/requirements.txt
```

3. Run the application:
```bash
streamlit run Files/interface.py
```

## 📊 Usage

### Main Dashboard
The main dashboard provides a 2x2 quadrant view:
- **Top-Left**: Stock Price with technical indicators
- **Top-Right**: DCF/Corporate Evaluation with scenario analysis
- **Bottom-Left**: Company Info and Analyst Estimates
- **Bottom-Right**: Forecast and Recommendations

### Navigation
- **Main Dashboard**: Overview of all analyzed stocks
- **Market Overview**: Portfolio-level metrics and distributions
- **Individual Analysis**: Detailed single-stock analysis
- **Comparison**: Side-by-side stock comparison

### Analysis Workflow
1. **Run Full Analysis**: Analyze all available stocks
2. **Individual Analysis**: Focus on specific stocks
3. **Scenario Analysis**: Explore Bear/Base/Bull DCF scenarios
4. **Sensitivity Analysis**: Test parameter sensitivity
5. **Export Results**: Download analysis data

## 🔧 Technical Details

### Data Sources
- **Price Data**: Yahoo Finance integration
- **Fundamental Data**: Financial statements and ratios
- **Analyst Data**: Price targets and ratings
- **Technical Indicators**: Calculated in real-time

### Analysis Methods
- **DCF Model**: Multi-stage growth model with terminal value
- **Technical Analysis**: RSI, MACD, Bollinger Bands, ATR
- **Risk Metrics**: Beta, volatility, signal strength
- **Scenario Modeling**: Monte Carlo-style sensitivity analysis

### Performance Features
- **Caching**: Intelligent data caching for faster analysis
- **Parallel Processing**: Multi-threaded analysis execution
- **Memory Management**: Optimized data handling
- **Real-time Updates**: Live data refresh capabilities

## 📈 Key Metrics

### Valuation Metrics
- Enterprise Value (EV)
- Equity Value
- Price per Share (DCF)
- WACC and Terminal Growth
- Scenario-based projections

### Technical Metrics
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands position
- ATR (Average True Range)
- Support/Resistance levels

### Risk Metrics
- Beta coefficient
- Signal strength (1-10)
- Risk level classification
- Technical confidence
- Overall recommendation

## 🎯 Use Cases

### Investment Research
- Fundamental analysis with DCF modeling
- Technical analysis for entry/exit timing
- Risk assessment and portfolio management
- Scenario planning and sensitivity analysis

### Portfolio Management
- Multi-stock comparison and analysis
- Signal-based investment decisions
- Risk-adjusted return evaluation
- Performance tracking and monitoring

### Educational
- Learn DCF valuation methodologies
- Understand technical analysis concepts
- Practice scenario modeling
- Explore financial data analysis

## 🔒 Security & Privacy

- **Data Privacy**: All analysis performed locally
- **No Data Storage**: No personal or sensitive data stored
- **Open Source**: Transparent codebase for review
- **Local Processing**: All calculations run on your machine

## 🤝 Contributing

This project is designed for educational and research purposes. Contributions are welcome for:
- Additional technical indicators
- Enhanced DCF modeling features
- Improved data visualization
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment choices.

## 📞 Support

For questions or issues:
- Review the documentation in the Files/ directory
- Check the Archives/ for additional resources
- Examine the code comments for implementation details

---

**Note**: This platform combines multiple analysis methodologies into a unified interface. The DCF valuations, technical signals, and analyst estimates should be used together to form a comprehensive investment thesis.

# Stock Analysis Interface

A comprehensive stock analysis platform featuring DCF valuation, technical analysis, and an interactive Streamlit dashboard.

## 🌐 **LIVE DEMO - Try It Now!**

**🎯 [**CLICK HERE TO TEST THE APPLICATION**](https://equity-valuation-interface.streamlit.app/)**

Experience the full functionality:
- 📊 **Interactive DCF Analysis** with Bear/Base/Bull scenarios
- 📈 **Real-time Technical Analysis** with advanced indicators
- 🎛️ **Professional Dashboard** with multiple views
- 💹 **Live Market Data** integration
- 📋 **Export Capabilities** for analysis results

*The live demo is deployed on Streamlit Cloud and updates automatically with the latest code.*

## 🚀 Local Installation (Optional)

If you prefer to run it locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Set up API keys for enhanced data (copy .env.example to .env)
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run interface.py
```

### 🔑 API Keys (Optional)
The application works without API keys using yfinance as the default data source. For enhanced data:
- **FMP API**: Get free key at [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
- **Polygon.io API**: Get free key at [Polygon.io](https://polygon.io/)

## 📊 Main Application

**`interface.py`** - Interactive Streamlit dashboard providing:
- Real-time stock analysis and visualization
- DCF valuation with Bear/Base/Bull scenarios
- Technical analysis with advanced indicators
- Professional dashboard with multiple views
- Export capabilities for analysis results

## 🔧 Core Components

### Analysis Engine
- **`analysis_combo.py`** - Main analysis orchestration and coordination
- **`enhanced_comprehensive_data.py`** - Data management and market data fetching
- **`analyst_estimates_data.py`** - Analyst price targets and ratings integration

### Valuation & Technical Analysis
- **`dcf_calculation.py`** - Complete DCF valuation implementation
- **`dcf_calibration.py`** - Industry-specific parameters and assumptions
- **`technical_analysis.py`** - RSI, MACD, Bollinger Bands, ATR indicators

### Data & Configuration
- **`financial_data.db`** - SQLite database for data storage
- **`requirements.txt`** - Python dependencies

## 📈 Features

### DCF Valuation
- Multi-stage growth projections
- WACC calculation and sensitivity analysis
- Terminal value modeling (perpetuity & exit multiple)
- Bear/Base/Bull scenario analysis

### Technical Analysis
- Real-time price charts with technical overlays
- RSI, MACD, Bollinger Bands, ATR indicators
- Support/Resistance level detection
- Risk management tools

### Dashboard Views
1. **Main Dashboard** - 2x2 quadrant overview
2. **Market Overview** - Portfolio-level analysis
3. **Individual Analysis** - Single-stock deep dive
4. **Comparison** - Multi-stock comparison

### Data Management
- Real-time market data via Yahoo Finance
- Analyst estimates integration
- Intelligent caching system
- SQLite database storage

## 🛠️ Dependencies

Key Python packages required:
- Streamlit (dashboard interface)
- Pandas & NumPy (data processing)
- Plotly (interactive charts)
- yfinance (market data)
- SQLite (data storage)

## 📁 File Structure

```
├── interface.py                 # Main Streamlit dashboard
├── analysis_combo.py           # Analysis orchestration
├── dcf_calculation.py          # DCF valuation engine
├── dcf_calibration.py          # Parameter calibration
├── technical_analysis.py       # Technical indicators
├── enhanced_comprehensive_data.py # Data management
├── analyst_estimates_data.py   # Analyst data integration
├── requirements.txt            # Dependencies
├── financial_data.db          # SQLite database
└── README.md                  # This file
```

## 🎯 Key Metrics

- **Enterprise Value (EV)**
- **Price per Share (DCF)**
- **Technical Signal Strength**
- **Risk Level Assessment**
- **Analyst Consensus**

## 🚀 Usage

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the application**: `streamlit run interface.py`
3. **Select stocks** from the sidebar
4. **Run analysis** using the control panel
5. **Explore scenarios** and sensitivity analysis
6. **Export results** as needed

## 🎯 **Ready to Test?**

**👉 [**LAUNCH THE LIVE APPLICATION NOW**](https://equity-valuation-interface.streamlit.app/)**

Try these features:
- 🔍 **Select any stock** from the dropdown
- 📊 **Run comprehensive analysis** with one click
- 🎲 **Explore Bear/Base/Bull scenarios** in DCF analysis
- 📈 **View real-time charts** with technical indicators
- ⚖️ **Compare multiple stocks** side-by-side
- 📋 **Export results** as CSV files

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Not intended as investment advice.

---

**Main Application**: Run `streamlit run interface.py` to start the dashboard locally, or **[test the live version](https://equity-valuation-interface.streamlit.app/)**
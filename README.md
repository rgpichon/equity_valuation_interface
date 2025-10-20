# Stock Analysis Interface

A comprehensive stock analysis platform featuring DCF valuation, technical analysis, and an interactive Streamlit dashboard.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run interface.py
```

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

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Not intended as investment advice.

---

**Main Application**: Run `streamlit run interface.py` to start the dashboard
# Stock Analysis Interface

A comprehensive stock analysis platform featuring DCF valuation, technical analysis, and an interactive Streamlit dashboard.

## 🚀 Main Application

**`interface.py`** - The main Streamlit dashboard application that provides:
- Interactive stock analysis interface
- DCF valuation with scenario modeling
- Technical analysis with advanced indicators
- Real-time data visualization
- Professional dashboard layout

## 📊 Core Features

### DCF Valuation Engine
- **`dcf_calculation.py`** - Complete DCF valuation implementation
- **`dcf_calibration.py`** - Parameter calibration and industry assumptions
- Multi-scenario analysis (Bear/Base/Bull)
- Sensitivity analysis and Monte Carlo modeling

### Technical Analysis
- **`technical_analysis.py`** - Advanced technical indicators
- RSI, MACD, Bollinger Bands, ATR
- Support/Resistance level detection
- Risk management tools

### Data Management
- **`enhanced_comprehensive_data.py`** - Centralized data management
- **`analyst_estimates_data.py`** - Analyst data integration
- Real-time market data fetching
- Intelligent caching system

### Analysis Orchestration
- **`analysis_combo.py`** - Main analysis coordination
- Portfolio-level analysis
- Signal generation and risk assessment
- Results management and export

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run interface.py
```

### Key Dependencies
- Streamlit (dashboard interface)
- Pandas & NumPy (data processing)
- Plotly (interactive charts)
- yfinance (market data)
- SQLite (data storage)

## 📈 Analysis Capabilities

### DCF Modeling
- Multi-stage growth projections
- WACC calculation and sensitivity
- Terminal value modeling
- Scenario-based valuations

### Technical Analysis
- Real-time price charts
- Technical indicator overlays
- Risk/reward analysis
- Signal strength assessment

### Portfolio Analysis
- Multi-stock comparison
- Signal distribution analysis
- Risk-adjusted metrics
- Performance tracking

## 🎯 Key Metrics

- **Enterprise Value (EV)**
- **Price per Share (DCF)**
- **Technical Signal Strength**
- **Risk Level Assessment**
- **Analyst Consensus**

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
└── analysis_results.*         # Cached results
```

## 🔧 Configuration

The application uses intelligent defaults but allows customization through:
- Industry-specific DCF assumptions
- Technical indicator parameters
- Risk assessment thresholds
- Data refresh intervals

## 📊 Dashboard Views

1. **Main Dashboard** - 2x2 quadrant overview
2. **Market Overview** - Portfolio-level analysis
3. **Individual Analysis** - Single-stock deep dive
4. **Comparison** - Multi-stock comparison

## 🚀 Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run interface.py`
3. Select stocks and run analysis
4. Explore scenarios and sensitivity
5. Export results as needed

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Not intended as investment advice.

---

**Main Application**: Run `streamlit run interface.py` to start the dashboard

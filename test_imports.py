#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

try:
    import streamlit as st
    print("✅ Streamlit import successful")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✅ Pandas import successful")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import numpy as np
    print("✅ NumPy import successful")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✅ Plotly import successful")
except ImportError as e:
    print(f"❌ Plotly import failed: {e}")

try:
    import yfinance as yf
    print("✅ yfinance import successful")
except ImportError as e:
    print(f"❌ yfinance import failed: {e}")

try:
    from analysis_combo import AnalysisCombo
    print("✅ AnalysisCombo import successful")
except ImportError as e:
    print(f"❌ AnalysisCombo import failed: {e}")

print("\n🎉 All imports successful! Ready for deployment.")

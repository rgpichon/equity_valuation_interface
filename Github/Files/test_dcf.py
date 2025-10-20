#!/usr/bin/env python3
"""Smoke tests for DCF calculation functions."""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcf_calculation import project_operating_model, value_with_perpetuity
from analysis_combo import run_scenario_analysis

def test_basic_functionality():
    """Test basic DCF functionality."""
    print("Testing DCF functionality...")
    
    # Test projection
    inputs = {
        'years': 5,
        'revenue_base': 100_000_000.0,
        'growth_path': [0.05] * 5,
        'ebitda_margin_now': 0.30,
        'margin_target': 0.35,
        'tax_rate': 0.22,
        'capex_pct_sales': 0.04,
        'nwc_pct_sales': 0.02,
        'depr_pct_sales': 0.03,
        'shares': 10_000_000.0,
        'net_debt': 0.0
    }
    
    proj = project_operating_model(inputs, {})
    assert isinstance(proj, pd.DataFrame), "Should return DataFrame"
    assert len(proj) == 5, "Should have 5 years"
    print("✅ Projection model works")
    
    # Test valuation
    adjustments = {'net_debt': 0.0, 'shares': 10_000_000.0}
    val = value_with_perpetuity(proj, wacc=0.10, terminal_g=0.03, adjustments=adjustments)
    assert 'EV' in val, "Should have EV"
    assert 'price_per_share' in val, "Should have price_per_share"
    print("✅ Valuation works")
    
    # Test scenario analysis
    scenario = {**inputs, 'wacc': 0.10, 'g': 0.03, 'mode': 'perpetuity'}
    result = run_scenario_analysis('TEST', scenario)
    assert 'valuation' in result, "Should have valuation"
    assert 'projection_df' in result, "Should have projection_df"
    print("✅ Scenario analysis works")
    
    print(f"Price per share: ${result['valuation']['price_per_share']:.2f}")
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

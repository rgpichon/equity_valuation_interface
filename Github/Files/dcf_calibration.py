"""
DCF Calibration Module
Robust, parameterized calibration for realistic WACC and DCF parameters.
Provides industry, country, size-based defaults with clean helpers.
"""

from typing import Dict, Optional, Tuple


# ==================== MODULE-LEVEL DEFAULTS ====================

INDUSTRY_DEFAULTS = {
    # Enhanced with volatility, cyclicality, and reinvestment intensity
    "Software": {
        "margin_target": 0.32, 
        "capex_pct_sales": 0.04, 
        "nwc_pct_sales": 0.02, 
        "tax_rate": 0.17, 
        "lt_growth_cap": 0.0275, 
        "exit_multiple_hint": 15.0,
        "industry_volatility": 0.25,  # Low volatility
        "cyclicality_factor": 1.0,    # Non-cyclical
        "reinvestment_intensity": 0.6  # Low CapEx/NWC
    },
    "Consumer Staples": {
        "margin_target": 0.20, 
        "capex_pct_sales": 0.04, 
        "nwc_pct_sales": 0.03, 
        "tax_rate": 0.22, 
        "lt_growth_cap": 0.0225, 
        "exit_multiple_hint": 12.0,
        "industry_volatility": 0.15,  # Very low volatility
        "cyclicality_factor": 0.8,    # Defensive
        "reinvestment_intensity": 0.7  # Low CapEx/NWC
    },
    "Semiconductors": {
        "margin_target": 0.30, 
        "capex_pct_sales": 0.07, 
        "nwc_pct_sales": 0.02, 
        "tax_rate": 0.17, 
        "lt_growth_cap": 0.0250, 
        "exit_multiple_hint": 13.0,
        "industry_volatility": 0.45,  # High volatility
        "cyclicality_factor": 1.4,    # Highly cyclical
        "reinvestment_intensity": 1.2  # High CapEx/NWC
    },
    "Hardware": {
        "margin_target": 0.22, 
        "capex_pct_sales": 0.05, 
        "nwc_pct_sales": 0.02, 
        "tax_rate": 0.17, 
        "lt_growth_cap": 0.0225, 
        "exit_multiple_hint": 12.0,
        "industry_volatility": 0.35,  # Medium-high volatility
        "cyclicality_factor": 1.2,    # Cyclical
        "reinvestment_intensity": 1.0  # Medium CapEx/NWC
    },
    "Biotech": {
        "margin_target": 0.18, 
        "capex_pct_sales": 0.03, 
        "nwc_pct_sales": 0.02, 
        "tax_rate": 0.18, 
        "lt_growth_cap": 0.0200, 
        "exit_multiple_hint": 10.0,
        "industry_volatility": 0.50,  # Very high volatility
        "cyclicality_factor": 1.1,    # Slightly cyclical
        "reinvestment_intensity": 0.5  # Low CapEx/NWC
    },
}

RISK_DEFAULTS = {
    # Enhanced anchors with more granular risk factors
    "risk_free_anchor": 0.043,                       # long-run 10y proxy
    "erp_anchor": {
        "developed": 0.050, 
        "emerging": 0.065, 
        "frontier": 0.080
    },
    "country_risk_premium": {
        "US": 0.000, 
        "CA": 0.001,  # Canada
        "GB": 0.002,  # UK
        "FR": 0.003,  # France
        "DE": 0.002,  # Germany
        "IT": 0.004,  # Italy
        "AU": 0.002,  # Australia
        "EU": 0.003, 
        "CN": 0.010, 
        "IN": 0.010, 
        "BR": 0.015, 
        "MX": 0.012, 
        "JP": 0.001
    },
    "size_premium": {
        "micro": 0.025,  # < $300M
        "small": 0.015,  # $300M - $2B
        "mid": 0.008,    # $2B - $10B
        "large": 0.004,  # $10B - $200B
        "mega": 0.002    # > $200B
    },
    "credit_spread": {
        "AAA": 0.008, 
        "AA": 0.010, 
        "A": 0.013, 
        "BBB": 0.018, 
        "BB": 0.030, 
        "B": 0.050,
        "CCC": 0.080,  # High yield
        "NR": 0.060    # Not rated
    },
    "statutory_tax_floor": 0.15,                     # global min floor
    "gdp_long_run": {
        "US": 0.020, 
        "CA": 0.018,  # Canada
        "GB": 0.016,  # UK
        "FR": 0.015,  # France
        "DE": 0.014,  # Germany
        "IT": 0.013,  # Italy
        "AU": 0.019,  # Australia
        "EU": 0.017, 
        "JP": 0.010, 
        "CN": 0.035, 
        "IN": 0.050, 
        "BR": 0.030,  # Brazil
        "MX": 0.025,  # Mexico
        "World": 0.025
    },
    "inflation_assumption": 0.020,  # Global inflation anchor
    "leverage_adjustment": 0.002,   # COE adjustment per D/E unit
    "emerging_market_multiplier": 1.25  # Credit spread multiplier for EM
}


# ==================== UTILITY FUNCTIONS ====================

def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a value between lower and upper bounds.
    
    Args:
        x: Value to clamp
        lo: Lower bound
        hi: Upper bound
        
    Returns:
        Clamped value
    """
    return float(min(max(x, lo), hi))


# ==================== SAFE PICKERS ====================

def pick_industry_defaults(industry_text: Optional[str]) -> Tuple[str, Dict]:
    """
    Pick industry defaults using substring matching.
    
    Args:
        industry_text: Industry description string
        
    Returns:
        Tuple of (matched_industry_name, parameters_dict)
    """
    if not industry_text:
        # Return first entry as fallback
        first_key = next(iter(INDUSTRY_DEFAULTS))
        return first_key, INDUSTRY_DEFAULTS[first_key]
    
    industry_text_lower = industry_text.lower()
    
    # Try exact match first
    for industry, params in INDUSTRY_DEFAULTS.items():
        if industry.lower() == industry_text_lower:
            return industry, params
    
    # Try substring match
    for industry, params in INDUSTRY_DEFAULTS.items():
        if industry.lower() in industry_text_lower or industry_text_lower in industry.lower():
            return industry, params
    
    # Fallback to first entry
    first_key = next(iter(INDUSTRY_DEFAULTS))
    return first_key, INDUSTRY_DEFAULTS[first_key]


def pick_region(country_code: Optional[str]) -> str:
    """
    Pick region from country code with enhanced mapping.
    
    Args:
        country_code: Country code string
            
        Returns:
        Region key present in country_risk_premium
    """
    if not country_code:
        return "World"  # Changed default to World
    
    country_upper = country_code.upper()
    
    # Check if country code exists in country_risk_premium
    if country_upper in RISK_DEFAULTS["country_risk_premium"]:
        return country_upper
    
    # Enhanced country mapping
    country_mapping = {
        "CA": "CA", "GB": "GB", "FR": "FR", "DE": "DE", "IT": "IT", "AU": "AU",
        "CN": "CN", "IN": "IN", "BR": "BR", "MX": "MX", "JP": "JP",
        "US": "US", "EU": "EU"
    }
    
    if country_upper in country_mapping:
        return country_mapping[country_upper]
    
    # Default to World for unmapped countries
    return "World"


def pick_size_bucket(market_cap_usd: Optional[float]) -> str:
    """
    Pick size bucket based on market cap with enhanced micro category.
    
    Args:
        market_cap_usd: Market capitalization in USD
            
        Returns:
        Size bucket string
    """
    if market_cap_usd is None:
        return "large"
    
    if market_cap_usd >= 200_000_000_000:  # $200B+
        return "mega"
    elif market_cap_usd >= 10_000_000_000:  # $10B+
        return "large"
    elif market_cap_usd >= 2_000_000_000:   # $2B+
        return "mid"
    elif market_cap_usd >= 300_000_000:     # $300M+
        return "small"
    else:
        return "micro"  # < $300M


# ==================== COST HELPERS ====================

def build_cost_of_equity(beta: Optional[float], country: Optional[str] = None, size_bucket: Optional[str] = None, debt_to_equity: float = 0.0) -> float:
    """
    Build cost of equity using enhanced CAPM with country, size, and leverage adjustments.
    
    Args:
        beta: Equity beta (defaults to 1.0)
        country: Country code for risk premium
        size_bucket: Size bucket for size premium
        debt_to_equity: Debt to equity ratio for leverage adjustment
            
        Returns:
        Cost of equity (clamped 5%-20%)
    """
    rf = RISK_DEFAULTS["risk_free_anchor"]
    
    # Enhanced ERP selection based on region
    region = pick_region(country)
    if region in ["CN", "IN", "BR", "MX"]:
        erp = RISK_DEFAULTS["erp_anchor"]["emerging"]
    elif region in ["US", "CA", "GB", "FR", "DE", "IT", "AU", "EU", "JP"]:
        erp = RISK_DEFAULTS["erp_anchor"]["developed"]
    else:
        erp = RISK_DEFAULTS["erp_anchor"]["frontier"]
    
    crp = RISK_DEFAULTS["country_risk_premium"].get(region, 0.0)
    sp = RISK_DEFAULTS["size_premium"].get(size_bucket or "large", 0.004)
    
    # Enhanced beta impact with more dispersion
    b = 1.0 if beta is None else float(beta)
    beta_adjustment = (b - 1.0) * 0.5  # More significant beta impact
    
    # Leverage adjustment
    leverage_adjustment = RISK_DEFAULTS["leverage_adjustment"] * debt_to_equity
    
    # Size premium enhancement for small/mid caps
    if size_bucket in ["small", "micro"]:
        sp *= 1.5  # Increase size premium weight for smaller firms
    
    coe = rf + b * (erp + crp) + sp + beta_adjustment + leverage_adjustment
    return clamp(coe, 0.05, 0.20)


def build_cost_of_debt(rating: Optional[str], country: Optional[str] = None) -> float:
    """
    Build cost of debt using risk-free rate plus credit spread with country adjustments.
    
    Args:
        rating: Credit rating (defaults to 'A')
        country: Country code for emerging market adjustment
            
        Returns:
        Pre-tax cost of debt (clamped 3%-15%)
    """
    rf = RISK_DEFAULTS["risk_free_anchor"]
    spr = RISK_DEFAULTS["credit_spread"].get((rating or "A").upper(), 0.013)
    
    # Country adjustment for emerging markets
    region = pick_region(country)
    if region in ["CN", "IN", "BR", "MX"]:
        spr *= RISK_DEFAULTS["emerging_market_multiplier"]
    
    # Minor country adjustment via CRP multiplier
    crp = RISK_DEFAULTS["country_risk_premium"].get(region, 0.0)
    country_adjustment = spr * (1 + crp * 0.1)  # Small CRP impact on credit spread
    
    return clamp(rf + spr + country_adjustment, 0.03, 0.15)


def after_tax_rate(statutory: Optional[float], country: Optional[str] = None) -> float:
    """
    Calculate after-tax rate with enhanced blended formula and country adjustments.
    
    Args:
        statutory: Statutory tax rate
        country: Country code for emerging market adjustment
        
    Returns:
        After-tax rate (clamped 15%-30%)
    """
    floor = RISK_DEFAULTS["statutory_tax_floor"]
    if statutory is None:
        return floor
    
    # Enhanced blended formula
    base_rate = clamp(float(statutory), floor, 0.30)
    
    # Emerging market adjustment (slightly lower effective taxes)
    region = pick_region(country)
    is_emerging = region in ["CN", "IN", "BR", "MX"]
    emerging_adjustment = 0.1 if is_emerging else 0.0
    
    effective_rate = base_rate * (1 - emerging_adjustment)
    return clamp(effective_rate, floor, 0.30)


def build_wacc(coe: float, cod_pre_tax: float, tax_rate: float, weight_equity: float, weight_debt: float, terminal_g: Optional[float] = None, size_bucket: Optional[str] = None, industry_volatility: float = 0.25) -> float:
    """
    Build WACC from cost components and weights with enhanced adaptive clamping.
    
    Args:
        coe: Cost of equity
        cod_pre_tax: Pre-tax cost of debt
        tax_rate: Tax rate
        weight_equity: Equity weight (should sum to 1.0 with debt weight)
        weight_debt: Debt weight
        terminal_g: Terminal growth rate (for validation)
        size_bucket: Size bucket for adaptive clamping
        industry_volatility: Industry volatility factor
            
    Returns:
        WACC with adaptive clamping
        
    Raises:
        ValueError: If WACC <= terminal_g
    """
    cod = cod_pre_tax * (1.0 - tax_rate)
    wacc = weight_equity * coe + weight_debt * cod
    
    # Enhanced terminal growth validation with safety margin
    if terminal_g is not None and wacc <= terminal_g + 0.015:
        raise ValueError("WACC must exceed terminal growth by at least 1.5%.")
    
    # Adaptive clamping based on size and volatility
    if size_bucket in ["mega", "large"]:
        wacc_min, wacc_max = 0.055, 0.105  # Large/mega firms: 5.5-10.5%
    else:
        wacc_min, wacc_max = 0.070, 0.140  # Mid/small firms: 7-14%
    
    # Industry volatility adjustment
    volatility_adjustment = (industry_volatility - 0.25) * 0.01  # ±1% for high/low volatility
    wacc += volatility_adjustment
    
    return clamp(wacc, wacc_min, wacc_max)


def validate_terminal_growth(g: float, industry_text: Optional[str] = None, country: Optional[str] = None, wacc: Optional[float] = None) -> float:
    """
    Validate and clamp terminal growth rate with enhanced dynamic validation.
    
    Args:
        g: Terminal growth rate to validate
        industry_text: Industry description for industry-specific cap
        country: Country code for country-specific cap
        wacc: WACC for safety margin validation
        
    Returns:
        Clamped terminal growth rate with dynamic caps
    """
    # Get industry cap
    industry_name, _ = pick_industry_defaults(industry_text)
    cap_ind = INDUSTRY_DEFAULTS.get(industry_name, {}).get("lt_growth_cap", 0.025)
    
    # Get country cap with inflation consideration
    country_region = pick_region(country)
    cap_geo = RISK_DEFAULTS["gdp_long_run"].get(country_region, 0.025)
    
    # Add inflation anchor for nominal growth
    inflation_anchor = RISK_DEFAULTS["inflation_assumption"]
    if cap_geo > 0.025:  # High growth regions
        cap_geo += inflation_anchor * 0.5  # Partial inflation adjustment
    
    # Regional caps
    if country_region in ["US", "CA", "GB", "FR", "DE", "IT", "AU", "EU", "JP"]:
        regional_cap = 0.0275  # Developed markets: 2.75%
    elif country_region in ["CN", "IN", "BR", "MX"]:
        regional_cap = 0.0400  # Emerging markets: 4.0%
    else:
        regional_cap = 0.0300  # World default: 3.0%
    
    # WACC safety margin
    wacc_cap = wacc - 0.015 if wacc else 0.030  # WACC - 1.5% safety margin
    
    # Apply all caps
    final_cap = min(cap_ind, cap_geo, regional_cap, wacc_cap)
    
    # Soft lower bound
    final_g = max(float(g), 0.005)  # Minimum 0.5% unless user overrides
    
    return clamp(final_g, 0.005, final_cap)


# ==================== CONVENIENCE FUNCTIONS ====================

def get_industry_parameters(industry_text: Optional[str]) -> Dict:
    """
    Get industry parameters with enhanced dynamic adjustments.
    
    Args:
        industry_text: Industry description
            
    Returns:
        Dictionary of industry parameters with dynamic adjustments
    """
    _, params = pick_industry_defaults(industry_text)
    
    # Create enhanced parameters with dynamic adjustments
    enhanced_params = params.copy()
    
    # Dynamic CapEx and NWC adjustments based on volatility
    volatility = params.get("industry_volatility", 0.25)
    cyclicality = params.get("cyclicality_factor", 1.0)
    reinvestment_intensity = params.get("reinvestment_intensity", 1.0)
    
    # Adjust CapEx and NWC within ±25% based on industry characteristics
    capex_base = params.get("capex_pct_sales", 0.05)
    nwc_base = params.get("nwc_pct_sales", 0.03)
    
    # Volatility adjustment
    volatility_multiplier = 1.0 + (volatility - 0.25) * 0.5  # ±25% adjustment
    enhanced_params["capex_pct_sales"] = clamp(capex_base * volatility_multiplier, capex_base * 0.75, capex_base * 1.25)
    enhanced_params["nwc_pct_sales"] = clamp(nwc_base * volatility_multiplier, nwc_base * 0.75, nwc_base * 1.25)
    
    # Reinvestment intensity adjustment
    enhanced_params["capex_pct_sales"] *= reinvestment_intensity
    enhanced_params["nwc_pct_sales"] *= reinvestment_intensity
    
    # Smooth margin targets (blend short-term and steady-state)
    margin_target = params.get("margin_target", 0.25)
    enhanced_params["margin_target_smooth"] = margin_target * 0.7 + margin_target * 0.3  # 70% steady-state, 30% short-term
    
    return enhanced_params


def build_company_wacc(beta: Optional[float], 
                      rating: Optional[str], 
                      market_cap: Optional[float], 
                      country: Optional[str], 
                      tax_rate: Optional[float],
                      debt_to_equity: float = 0.3,
                      terminal_g: Optional[float] = None,
                      industry_text: Optional[str] = None) -> Dict:
    """
    Build complete WACC for a company with enhanced calibration.
    
    Args:
        beta: Equity beta
        rating: Credit rating
        market_cap: Market capitalization in USD
        country: Country code
        tax_rate: Tax rate
        debt_to_equity: Debt to equity ratio (default 0.3)
        terminal_g: Terminal growth rate
        industry_text: Industry description for volatility adjustment
            
    Returns:
        Dictionary with WACC components and final WACC
    """
    # Get size bucket and region
    size_bucket = pick_size_bucket(market_cap)
    country_region = pick_region(country)
    
    # Get industry parameters for volatility
    industry_name, industry_params = pick_industry_defaults(industry_text)
    industry_volatility = industry_params.get("industry_volatility", 0.25)
    
    # Build costs with enhanced parameters
    coe = build_cost_of_equity(beta, country_region, size_bucket, debt_to_equity)
    cod_pre_tax = build_cost_of_debt(rating, country_region)
    effective_tax_rate = after_tax_rate(tax_rate, country_region)
    
    # Calculate weights (assuming market value weights)
    total_capital = 1.0 + debt_to_equity
    weight_equity = 1.0 / total_capital
    weight_debt = debt_to_equity / total_capital
    
    # Build WACC with enhanced parameters
    wacc = build_wacc(coe, cod_pre_tax, effective_tax_rate, weight_equity, weight_debt, terminal_g, size_bucket, industry_volatility)
    
    # Calculate inflation assumption
    inflation_assumption = RISK_DEFAULTS["gdp_long_run"].get(country_region, RISK_DEFAULTS["inflation_assumption"])
    
    # Calculate real vs nominal WACC
    real_wacc = wacc - inflation_assumption
    
    # Validate and suggest terminal growth
    suggested_terminal_g = validate_terminal_growth(terminal_g or 0.025, industry_text, country, wacc)
    
    return {
        "wacc": wacc,  # Nominal WACC (for DCF consistency)
        "cost_of_equity": coe,
        "cost_of_debt_pre_tax": cod_pre_tax,
        "cost_of_debt_after_tax": cod_pre_tax * (1 - effective_tax_rate),
        "tax_rate": effective_tax_rate,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "size_bucket": size_bucket,
        "country_region": country_region,
        "beta": beta or 1.0,
        "rating": rating or "A",
        # Optional enhanced fields
        "inflation_assumption": inflation_assumption,
        "nominal_wacc": wacc,
        "real_wacc": real_wacc,
        "terminal_g_suggested": suggested_terminal_g
    }


# ==================== EXPORTS ====================

__all__ = [
    'INDUSTRY_DEFAULTS',
    'RISK_DEFAULTS',
    'clamp',
    'pick_industry_defaults',
    'pick_region',
    'pick_size_bucket',
    'build_cost_of_equity',
    'build_cost_of_debt',
    'after_tax_rate',
    'build_wacc',
    'validate_terminal_growth',
    'get_industry_parameters',
    'build_company_wacc'
]
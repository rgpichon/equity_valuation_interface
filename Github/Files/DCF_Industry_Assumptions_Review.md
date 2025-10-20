# DCF Industry Assumptions Review & Calibration

## Your Stock Portfolio
- **AAPL** (Apple) - Technology/Hardware
- **BABA** (Alibaba) - E-commerce
- **CAT** (Caterpillar) - Manufacturing/Industrials
- **NVO** (Novo Nordisk) - Pharmaceuticals
- **SIEGY** (Siemens) - Manufacturing/Industrials/Conglomerate

---

## Industry-by-Industry Review

### 1. HARDWARE (Apple - AAPL)

**Current Assumptions:**
```
Beta: 1.25
Long-term Growth: 2.5%
EBITDA Margin: 15%
CapEx/Sales: 5%
NWC/Sales: 15%
Debt/Equity: 0.25
Tax Rate: 21%
```

**Real-World Benchmarks (Apple Actual):**
- **EBITDA Margin**: Apple's actual EBITDA margin is ~33-35% (significantly higher than our 15%)
- **Gross Margin**: 44-46% (premium brand)
- **CapEx/Sales**: 2-3% (lower than our 5% - asset-light model)
- **NWC/Sales**: Negative or near zero (efficient supply chain)
- **Beta**: ~1.2-1.3 (our 1.25 is accurate)
- **Terminal Growth**: 2.5% is reasonable for mature tech

**Recommended Adjustments:**
```python
'Hardware': IndustryParameters(
    beta=1.25,                      # Keep (accurate)
    long_term_growth=0.025,         # Keep (realistic for mature tech)
    ebitda_margin=0.30,             # INCREASE from 0.15 (Apple is premium)
    capex_sales_ratio=0.03,         # DECREASE from 0.05 (asset-light)
    nwc_sales_ratio=0.05,           # DECREASE from 0.15 (efficient)
    debt_to_equity=0.25,            # Keep
    tax_rate=0.15                   # DECREASE from 0.21 (Apple's effective rate ~15%)
)
```

**Rationale**: Apple operates a premium business model with high margins. Our generic "Hardware" assumptions were too conservative and didn't reflect Apple's pricing power and operational efficiency.

---

### 2. E-COMMERCE (Alibaba - BABA)

**Current Assumptions:**
```
Beta: 1.30
Long-term Growth: 3.5%
EBITDA Margin: 6%
CapEx/Sales: 6%
NWC/Sales: 8%
Debt/Equity: 0.30
Tax Rate: 21%
```

**Real-World Benchmarks (Alibaba/E-commerce):**
- **EBITDA Margin**: Alibaba core commerce ~20-25%, overall ~15% (much higher than our 6%)
- **Operating Margin**: 10-15% (marketplace model is capital-light)
- **CapEx/Sales**: 3-4% (mostly cloud infrastructure)
- **Beta**: 1.3-1.5 (high volatility, China risk)
- **Tax Rate**: 15-25% (China corporate tax, varies by entity)
- **Terminal Growth**: 3.5% might be high given China's slowing growth

**Recommended Adjustments:**
```python
'E-commerce': IndustryParameters(
    beta=1.35,                      # INCREASE from 1.30 (China risk premium)
    long_term_growth=0.030,         # DECREASE from 0.035 (China slowdown)
    ebitda_margin=0.15,             # INCREASE from 0.06 (platform economics)
    capex_sales_ratio=0.04,         # DECREASE from 0.06 (asset-light)
    nwc_sales_ratio=0.08,           # Keep (merchant receivables)
    debt_to_equity=0.30,            # Keep
    tax_rate=0.20                   # DECREASE from 0.21 (China effective rate)
)
```

**Rationale**: E-commerce platforms have much better margins than traditional retail (6% was way too low). The marketplace model is capital-light with high operating leverage.

---

### 3. MANUFACTURING (Caterpillar, Siemens - CAT, SIEGY)

**Current Assumptions:**
```
Beta: 1.10
Long-term Growth: 2.5%
EBITDA Margin: 10%
CapEx/Sales: 6%
NWC/Sales: 14%
Debt/Equity: 0.65
Tax Rate: 23%
```

**Real-World Benchmarks (CAT/SIEGY):**
- **Caterpillar EBITDA Margin**: 18-22% (actual, much higher than 10%)
- **Siemens EBITDA Margin**: 12-16% (diversified conglomerate)
- **CapEx/Sales**: 4-5% for CAT, 3-4% for Siemens
- **Beta**: 1.0-1.2 (cyclical but not extreme)
- **Debt/Equity**: 1.5-2.0 for CAT (uses leverage), 0.3-0.5 for Siemens
- **Terminal Growth**: 2.5% is reasonable

**Recommended Adjustments:**
```python
'Manufacturing': IndustryParameters(
    beta=1.10,                      # Keep (reasonable)
    long_term_growth=0.025,         # Keep (realistic)
    ebitda_margin=0.15,             # INCREASE from 0.10 (modern manufacturing)
    capex_sales_ratio=0.04,         # DECREASE from 0.06 (efficiency gains)
    nwc_sales_ratio=0.12,           # DECREASE from 0.14 (JIT inventory)
    debt_to_equity=0.80,            # INCREASE from 0.65 (CAT uses leverage)
    tax_rate=0.23                   # Keep
)
```

**Rationale**: Modern industrial companies (especially CAT) have improved margins significantly through productivity gains, pricing power, and services. Our 10% was too pessimistic.

---

### 4. PHARMACEUTICALS (Novo Nordisk - NVO)

**Current Assumptions:**
```
Beta: 0.90
Long-term Growth: 3.0%
EBITDA Margin: 30%
CapEx/Sales: 4%
NWC/Sales: 15%
Debt/Equity: 0.35
Tax Rate: 15%
```

**Real-World Benchmarks (Novo Nordisk):**
- **EBITDA Margin**: Novo Nordisk ~45-50% (blockbuster drug margins, higher than 30%)
- **Operating Margin**: 40-45% (Ozempic/Wegovy premium pricing)
- **R&D/Sales**: 13-15% (not just CapEx - critical for pharma)
- **CapEx/Sales**: 5-6% (manufacturing facilities)
- **Beta**: 0.7-0.9 (defensive, healthcare)
- **Tax Rate**: 18-22% (Denmark/global blended)
- **Terminal Growth**: 3.0% is reasonable (demographics + innovation)

**Recommended Adjustments:**
```python
'Pharmaceuticals': IndustryParameters(
    beta=0.85,                      # DECREASE from 0.90 (more defensive)
    long_term_growth=0.030,         # Keep (aging demographics support)
    ebitda_margin=0.40,             # INCREASE from 0.30 (blockbuster drugs)
    capex_sales_ratio=0.05,         # INCREASE from 0.04 (facility investment)
    nwc_sales_ratio=0.12,           # DECREASE from 0.15 (efficient)
    debt_to_equity=0.25,            # DECREASE from 0.35 (conservative balance sheet)
    tax_rate=0.20                   # INCREASE from 0.15 (realistic effective rate)
)
```

**Rationale**: Companies with blockbuster drugs (like Novo's GLP-1 franchise) achieve much higher margins than generic pharma. Need to reflect this premium quality.

---

## Summary of Recommended Changes

### Critical Adjustments (High Impact):

1. **Hardware EBITDA Margin**: 15% → 30% (Apple's premium model)
2. **Hardware Tax Rate**: 21% → 15% (Apple's effective rate)
3. **E-commerce EBITDA Margin**: 6% → 15% (platform economics)
4. **Manufacturing EBITDA Margin**: 10% → 15% (modern productivity)
5. **Pharmaceuticals EBITDA Margin**: 30% → 40% (blockbuster drugs)

### Impact on DCF Valuations:
- **Higher EBITDA margins** → Higher cash flows → **Higher valuations** (20-40% increase)
- **Lower CapEx ratios** → More FCF → **Higher valuations** (5-10% increase)
- **Lower tax rates** → Higher after-tax cash → **Higher valuations** (5-8% increase)

**Net Effect**: Our current assumptions were **systematically undervaluing** quality companies. These adjustments should improve DCF accuracy significantly.

---

## Implementation Plan

1. **Update dcf_calibration.py** with revised industry parameters
2. **Add company-specific override capability** for exceptional companies (Apple, Novo)
3. **Test on your 5 stocks** to validate improvements
4. **Compare DCF outputs** before/after to quantify impact

Would you like me to implement these changes now?


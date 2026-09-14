---
name: stock-market-skill
description: Identify and analyze stocks with the highest percentage increase (top gainers) or lowest percentage decrease / greatest decline (top losers) using market data and registry flat-files.
---

# Stock Market Screener Skill

This skill enables the autonomous agent to answer user queries regarding equity market momentum, identifying stocks exhibiting either the **highest percentage increase** (top gainers) or the **lowest percentage decrease** (greatest drop / top losers).

## 🎯 Trigger Queries & User Intents
Activate this skill when the user asks:
- "Which stocks have the highest percentage increase?"
- "Which stocks have the lowest percentage decrease / biggest drop?"
- "Show me today's top stock gainers."
- "What stocks lost the most value?"
- "What are the biggest market movers today?"

---

## 📂 Architecture & Data Assets

Following the `SKILL_SPEC.md` directory layout:
```
skills/stock-market-skill/
├── SKILL.md                 # Metadata & Standard Operating Procedure (SOP)
├── data/
│   └── registry.csv         # Flat-file database for record resolution & offline fallback
└── scripts/
    └── stock_tools.py       # Performance calculator & public screener integration
```

### 1. Flat-File Registry (`data/registry.csv`)
Stores curated equity records including `symbol`, `name`, `sector`, `previous_close`, `current_price`, `change_percent`, `volume`, and `market_cap_b`. Used for fast local resolution and zero-external-dependency fallback.

### 2. Live Screener Integration (`scripts/stock_tools.py`)
Utilizes free public market screener feeds (`day_gainers` and `day_losers`) without requiring any paid API keys. Automatically falls back to `data/registry.csv` if network connectivity is interrupted.

---

## 🧮 Mathematical & Metric Definitions

1. **Percentage Change Formula:**
   $$\text{Change Percentage (\%)} = \left(\frac{\text{Current Price} - \text{Previous Close}}{\text{Previous Close}}\right) \times 100$$

2. **Highest Percentage Increase (Gainers):**
   - Stocks sorted in **descending order** by $\text{Change Percentage}$ (e.g., $+15.81\%$, $+15.80\%$, $+15.48\%$).
   - Represents assets appreciating most sharply in the current trading period.

3. **Lowest Percentage Decrease (Losers / Drops):**
   - Stocks sorted in **ascending order** by $\text{Change Percentage}$ (e.g., $-19.67\%$, $-13.01\%$, $-12.39\%$).
   - Note: In financial terminology, "lowest percentage decrease" or "biggest loss" refers to the equities with the most negative percentage movement.

---

## 📋 Standard Operating Procedure (SOP)

When answering questions about highest percentage increase or lowest percentage decrease:
1. **Determine the Metric Requested**:
   - For **highest percentage increase / gainers / best performers**: use `--gainers`.
   - For **lowest percentage decrease / biggest drops / losers**: use `--losers`.
2. **Execute the Performance Tool**:
   Run the script from the repository root:
   ```bash
   # For highest percentage increase:
   python skills/stock-market-skill/scripts/stock_tools.py --gainers --limit 5

   # For lowest percentage decrease / biggest drop:
   python skills/stock-market-skill/scripts/stock_tools.py --losers --limit 5
   ```
3. **Synthesize the Response**:
   - Provide a clear, formatted table containing:
     - **Ticker Symbol**
     - **Company Name**
     - **Current Price ($)**
     - **Change Percentage (%)**
     - **Data Source** (e.g., Live Screener or Local Registry)
   - Add a brief market insight summarizing why these equities are leading or lagging.

---

## 💻 CLI Commands & Examples

```bash
# Top 5 gainers
python skills/stock-market-skill/scripts/stock_tools.py --gainers

# Top 5 losers
python skills/stock-market-skill/scripts/stock_tools.py --losers

# Output JSON with custom limit
python skills/stock-market-skill/scripts/stock_tools.py --gainers --limit 10 --json
```

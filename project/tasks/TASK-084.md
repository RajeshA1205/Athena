# TASK-084: Results display — recommendation cards

**Sprint:** 11 — React Frontend
**Priority:** 🟡 High
**Status:** Queued
**Dependencies:** TASK-083

## Description
Build the results panel shown after analysis completes. Displays the coordinator's final recommendation, market analyst summary, risk assessment, and strategy signals in structured cards.

## Components

### `frontend/src/components/results/ResultPanel.tsx`
Top-level container, shown when `result` from `useAnalysis` is non-null.

### `frontend/src/components/results/RecommendationCard.tsx`
The hero card:
- Large BUY / SELL / HOLD badge (color coded)
- Confidence percentage
- Symbol + timestamp
- Coordinator reasoning text

### `frontend/src/components/results/MarketCard.tsx`
- Market regime (trending/ranging/volatile)
- RSI value with color coding (overbought/oversold)
- Trend direction

### `frontend/src/components/results/RiskCard.tsx`
- Risk level badge (LOW/MEDIUM/HIGH)
- VaR (95%) value
- Max drawdown
- Compliance violations list (if any)

### `frontend/src/components/results/SignalsCard.tsx`
- Table of strategy signals: type, strength, timestamp
- Each row color coded by signal type (BUY=green, SELL=red, HOLD=amber)

## Data Shape (from POST /analyze result event)
```typescript
interface AnalysisResult {
  coordinator: { action: string; confidence: number; risk_level: string; reasoning: string }
  market: { regime: string; rsi: number; trend: string }
  risk: { risk_level: string; var_95: number; max_drawdown: number }
  signals: Array<{ type: string; strength: number; timestamp: string }>
}
```

## Acceptance Criteria
- All four cards render correctly with real API data
- BUY/SELL/HOLD badge uses correct color
- Cards animate in on first render (fade-up, 150ms stagger)
- Empty/null fields display gracefully (em dash or "N/A")

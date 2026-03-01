# TASK-081: Design system + dragon SVG logo

**Sprint:** 11 — React Frontend
**Priority:** 🔴 Critical
**Status:** Queued
**Dependencies:** TASK-079

## Description
Create the ATHENA design system components and the dragon SVG logo. The aesthetic mirrors melboucierayane.com: brutalist minimalism, dark near-black background, Geist font, subtle card borders, monospace accents for data.

## Dragon Logo
- SVG dragon silhouette, abstract/geometric style (not cartoonish)
- Fits in a 40×40px header slot
- Scales cleanly, single color (`currentColor`) so it respects theme
- Save as `frontend/src/components/DragonLogo.tsx` (inline SVG as React component)

## Design Tokens (globals.css CSS variables)
```css
--color-bg: #0a0a0a;
--color-surface: #111111;
--color-border: #1f1f1f;
--color-fg: #fafafa;
--color-fg-muted: #71717a;
--color-accent: #22c55e;   /* green — financial positive */
--color-danger: #ef4444;   /* red — negative/sell */
--color-warn: #f59e0b;     /* amber — hold/caution */
--color-mono: 'GeistMono', monospace;
```

## Base Components (in `frontend/src/components/ui/`)
- `Card.tsx` — dark surface card with border, optional hover glow
- `Badge.tsx` — small pill for BUY/SELL/HOLD with color coding
- `Spinner.tsx` — minimal animated loading indicator
- `Divider.tsx` — hairline separator

## Acceptance Criteria
- DragonLogo renders without errors
- CSS variables applied globally
- All ui/ components export correctly with TypeScript props
- No Tailwind `@apply` in .tsx files (use cn() utility from shadcn pattern)

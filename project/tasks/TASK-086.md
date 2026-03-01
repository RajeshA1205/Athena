# TASK-086: Polish — animations, responsive, dark theme refinement

**Sprint:** 11 — React Frontend
**Priority:** 🟢 Medium
**Status:** Queued
**Dependencies:** TASK-083, TASK-084, TASK-085

## Description
Final polish pass: smooth animations, mobile responsiveness, consistent dark theme, and UX improvements. Goal: match the refined minimalist feel of melboucierayane.com.

## Animations
- Hero section: fade-in on load (opacity 0→1, translateY 10px→0, 500ms ease-out)
- Result cards: staggered fade-up (150ms delay per card)
- Agent stream rows: slide-in from left as they appear
- Sidebar active highlight: smooth transition (200ms)
- Badge: subtle pulse for HOLD, no animation for BUY/SELL

## Responsive Breakpoints
- Mobile (<768px): sidebar hidden, hamburger menu top-right, full-width cards
- Tablet (768–1024px): sidebar 200px, main content fills rest
- Desktop (>1024px): sidebar 240px, main content max-w-3xl centered

## Dark Theme Refinements
- Ensure no white backgrounds appear on any element
- Code/monospace text uses `--color-fg-muted` not full white
- Focus rings use `--color-accent` (green)
- Scrollbar styled: thin, dark track, muted thumb

## UX Improvements
- Symbol input shows example placeholder: "e.g. TSLA, AAPL, NVDA"
- Previous analyses listed in a history dropdown (stored in localStorage, last 10)
- Copy button on result cards to copy recommendation as text
- Keyboard shortcut `Cmd+K` focuses symbol input from anywhere

## Acceptance Criteria
- Lighthouse performance score ≥ 80 on mobile
- No layout shift (CLS < 0.1)
- All interactive elements have focus styles
- `npm run build` produces no TypeScript errors or ESLint warnings

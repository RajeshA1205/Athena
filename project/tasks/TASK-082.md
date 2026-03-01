# TASK-082: Core layout — shell, header, sidebar nav

**Sprint:** 11 — React Frontend
**Priority:** 🟡 High
**Status:** Queued
**Dependencies:** TASK-081

## Description
Build the main application shell: sticky header with dragon logo + ATHENA branding, a scroll-following sidebar for section navigation, and the single-page layout structure.

## Layout Structure
```
┌─────────────────────────────────────────────┐
│  🐉 ATHENA          AI Financial Advisor     │  ← sticky header, h-14
├──────────┬──────────────────────────────────┤
│ Sidebar  │  Main content area               │
│ (fixed   │  (scrollable)                    │
│  left)   │                                  │
│ • Analyze│                                  │
│ • Memory │                                  │
│ • Stats  │                                  │
│ • About  │                                  │
└──────────┴──────────────────────────────────┘
```

## Components
- `frontend/src/components/layout/Shell.tsx` — root layout wrapper
- `frontend/src/components/layout/Header.tsx` — sticky top bar with DragonLogo + "ATHENA" wordmark + subtitle "AI Financial Advisor"
- `frontend/src/components/layout/Sidebar.tsx` — fixed left nav with section links, highlights active section on scroll
- `frontend/src/app/page.tsx` — main page with sections: Hero, Analyze, Memory, Stats

## Hero Section
Short tagline: "Intelligence that remembers." followed by sub-copy explaining the 5-agent system. Minimal, centered, large typography. No image/3D needed.

## Acceptance Criteria
- Sidebar highlights correct section as user scrolls
- Header stays sticky on scroll
- DragonLogo visible in header
- Responsive: sidebar collapses to hamburger on mobile
- `npm run build` passes

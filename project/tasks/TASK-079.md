# TASK-079: Scaffold Next.js 14 frontend project

**Sprint:** 11 — React Frontend
**Priority:** 🔴 Critical
**Status:** Queued
**Dependencies:** None

## Description
Initialize the Next.js 14 project in `/Users/rajesh/athena/frontend/` using the App Router, TypeScript, and Tailwind CSS v3. Set up Geist font, configure dark theme CSS variables, and establish the project structure.

## Acceptance Criteria
- `frontend/` created with `npx create-next-app@14` (App Router, TypeScript, Tailwind, ESLint)
- Geist font configured via `next/font/google`
- CSS variables for dark theme: `--bg: #0a0a0a`, `--fg: #fafafa`, `--gray-*` scale
- `tsconfig.json` path alias `@/*` → `src/*`
- `npm run build` succeeds with no errors
- `npm run dev` starts on port 3000

## Files
- `frontend/` (new directory, entire scaffold)
- `frontend/src/app/layout.tsx`
- `frontend/src/app/globals.css`
- `frontend/tailwind.config.ts`
- `frontend/.env.local.example` with `NEXT_PUBLIC_API_URL=http://localhost:8000`

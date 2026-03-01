# TASK-092: Add Ingest Scheduler

## Status
- **State:** Queued
- **Priority:** 🟢 Medium
- **Depends on:** TASK-089
- **Created:** 2026-03-01

## Objective
Create `ingest/run_scheduler.py` — a standalone script that runs `collect_data.py` on a configurable schedule (default: once daily at 17:00 US/Eastern, after US market close). The scheduler must log each run to `logs/ingest_scheduler.log`.

## Context
The ingest pipeline (`ingest/collect_data.py`) is currently a one-shot script. To keep `data/market/` fresh, it needs to run automatically after each trading day. This script is meant to be started once and left running (e.g., via `python ingest/run_scheduler.py` or as a system service).

17:00 ET is chosen because US equities markets close at 16:00 ET and YFinance daily bars are typically available within the hour.

Project root: `/Users/rajesh/athena/`

## Scope & Constraints
- **May create:** `ingest/run_scheduler.py`
- **May NOT modify:** `ingest/collect_data.py`, any other existing file
- Use `schedule` library (preferred, lightweight) or `apscheduler` — whichever is already installed; check with `import schedule` before `import apscheduler`
- If neither is installed, implement using a simple `time.sleep` loop with a next-run calculation (no new dependencies required)
- The scheduled time must be configurable via a `--time` CLI argument (default `"17:00"`) and timezone via `--tz` (default `"US/Eastern"`)
- Log file path: `logs/ingest_scheduler.log` (relative to project root); create `logs/` dir if missing
- Each run must log: start time, symbols collected, duration, any errors
- The script must handle `KeyboardInterrupt` gracefully (log shutdown message, exit cleanly)

## Input
- `ingest/collect_data.py` — the script to invoke on schedule
- `ingest/config.yaml` — to understand what collect_data.py expects

## Expected Output
`ingest/run_scheduler.py` with:
1. `main()` entry point with `argparse` for `--time` and `--tz`
2. Scheduler loop that calls `collect_data` logic (or subprocess) at the configured time
3. Logging to both console (INFO) and `logs/ingest_scheduler.log` (INFO)
4. Graceful `KeyboardInterrupt` handler

## Acceptance Criteria
- [ ] `ingest/run_scheduler.py` exists and is syntactically valid (`python3 -m py_compile ingest/run_scheduler.py` succeeds)
- [ ] Script accepts `--time` argument (default `"17:00"`)
- [ ] Script accepts `--tz` argument (default `"US/Eastern"`)
- [ ] Logging goes to `logs/ingest_scheduler.log` (file created automatically)
- [ ] `logs/` directory is created automatically if missing
- [ ] Each scheduled run logs start time, completion status, and duration
- [ ] `KeyboardInterrupt` exits cleanly with a logged shutdown message
- [ ] No hardcoded absolute paths (use `pathlib.Path(__file__).parent.parent` for project root)
- [ ] `pytest tests/ -q` still passes (173 passed, 4 skipped baseline — this file has no tests, just verify no import breakage)

## Agent Log
| Date | Agent | Action | Result |
|------|-------|--------|--------|
| | | | |

## Review Notes
(Filled in during review)

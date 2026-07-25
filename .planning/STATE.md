---
status: active
phase: 1
plan: null
progress: 0/4 phases complete
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-07-22)

**Core value:** Produce auditable Qwen3-VL results whose inputs, candidate-action protocol, and failures can be independently reproduced without retraining a local model.

**Current focus:** Phase 1 — Audit and visual manifests.

## Current Position

Phase: 1 of 4 (Audit and visual manifests)  
Status: Table V frame extraction and local manifest tooling implemented; Phase 1 awaits source-video access and extracted observations  
Last activity: 2026-07-25 — added the Table V T=4, 3+3 extractor, portable manifest builder, strict legacy-action parser, offline Qwen configuration, and tests.

## Accumulated Context

### Decisions

- Create the separate `oepp_api_eval/` workspace; do not change the original OEPP source/data paths.
- Use `uv` and its committed `uv.lock`; do not create or use a Conda environment.
- Treat image availability as a hard manifest prerequisite. Current annotation JSON contains action segments and timestamps, but no raw image paths.
- Use exact original action text and record seeded candidate ordering.
- Preserve invalid model outputs as evaluation failures; no semantic repair.
- Use the historical Table V protocol as a separate `table_v_t4_3x3_legacy_v1` run: $T=4$, 3+3 observations, Base→Base and Novel→Novel pools, strict numbered action-name parsing.
- Extract frames on the video server with this fork, then synchronize frame files and relative-path observation indices to the local API workspace.

### Evidence

- The local split and segment counts reproduce Table II exactly: train 1,285/5,833; validation 337/1,479; base test 416/1,888; novel test 733/3,010.
- Base, Novel, and Total pools contain 122, 55, and 161 unique actions. Base and Novel overlap by 16 strings.
- The checkout contains only `img/intro-sample.png`; no raw test frames or videos are available.
- MinerU recovered the double-column paper and tables from `OEPP-TIP.pdf` successfully.
- The Table V sequence files match local annotations uniquely for all 888 Base and 1,297 Novel windows. `uv sync`, changed-file Ruff checks, 14 unit tests, and the disabled API gate passed.

### Blockers/Concerns

- **Visual-data blocker:** the raw MP4s remain external. Run `extract_tablev_frames.py` against the server's Table V sequence files, record unavailable videos, then synchronize the frames and observation JSONL before constructing a local manifest.

## Session Continuity

Last session: 2026-07-25  
Stopped at: Offline API implementation is ready; extract source-backed Table V observations on the server, synchronize them locally, then run the approved API pilot.  
Resume file: `.planning/STATE.md`

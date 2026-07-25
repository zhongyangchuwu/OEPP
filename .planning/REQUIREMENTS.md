# Requirements: OEPP Qwen3-VL API Evaluation

**Defined:** 2026-07-22  
**Core Value:** Produce auditable Qwen3-VL results whose inputs, candidate-action protocol, and failures can be independently reproduced without retraining a local model.

## Current Requirements

### Data and Manifest

- [x] **DATA-01**: A researcher can generate `data_audit.json` and `data_audit.md` that enumerate OEPP data files, split counts, action-pool sizes, horizon counts, duplicate identifiers, pool-membership violations, and visual-asset availability.
- [ ] **DATA-02**: A researcher can build JSONL manifests only from annotation windows with validated local start and goal frame files, exact ground-truth action strings, sequential candidate IDs, and a recorded candidate-order seed.
- [ ] **DATA-03**: A researcher can select reproducible Base and Novel pilot subsets without mutating the original annotation files.

### Remote API Safety

- [ ] **SAFE-01**: A researcher can configure Qwen3-VL endpoint, model ID, and API key solely through ignored environment variables.
- [ ] **SAFE-02**: The initial project state cannot issue an API request; later execution is capped, serialized for pilots, resumable, and logs no API key or image base64 payload.
- [ ] **API-01**: Each actual request saves non-secret request metadata, complete raw response, usage, latency, model ID, prompt version, candidate protocol, candidate order, parse result, and retry count.

### Evaluation and Reporting

- [x] **EVAL-01**: Parsed predictions are accepted only when they contain exactly `T` candidate IDs; malformed, missing, and out-of-pool outputs are recorded as failures rather than repaired.
- [x] **EVAL-02**: Evaluation computes paper-compatible SR, per-step Acc, and mean set IoU with failed predictions remaining in the denominator, plus macro-by-event statistics and API/parse failure rates.
- [ ] **EXP-01**: The planned study compares Base and Novel at $T=3,4$, 1+1 and 3+3 observations, split-specific and Total pools, and fixed candidate-order seeds after pilot approval.

## Deferred

- Cost-normalized comparison across providers — requires actual provider pricing and approved results.
- A raw-video frame extraction pipeline — requires source video locations and provenance that are absent from this checkout.

## Out of Scope

| Feature | Reason |
|---|---|
| Training or local inference | The evaluation is remote-API-only. |
| Editing OEPP model/data code | The original baseline must remain unchanged. |
| Automatic semantic correction of predictions | It would bias results and invalidate the protocol. |
| Full API sweep before review | A manual approval gate is required for cost and data-quality control. |

## Traceability

| Requirement | Phase | Status |
|---|---:|---|
| DATA-01 | Phase 1 | Complete — audit generated from local data |
| DATA-02 | Phase 1 | Blocked — source-backed visual observations absent |
| DATA-03 | Phase 1 | Blocked — depends on DATA-02 |
| SAFE-01 | Phase 2 | Implemented — awaits private environment values |
| SAFE-02 | Phase 2 | Implemented — offline gate verified; remote path unverified |
| API-01 | Phase 2 | Implemented — remote path unverified |
| EVAL-01 | Phase 2 | Complete — parser regression tests pass |
| EVAL-02 | Phase 2 | Complete — evaluator regression tests pass |
| EXP-01 | Phase 3 | Pending |

**Coverage:** 9 current requirements; 9 mapped; 0 unmapped.

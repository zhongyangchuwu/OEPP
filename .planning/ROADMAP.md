# Roadmap: OEPP Qwen3-VL API Evaluation

## Overview

Establish an audited visual-data manifest before enabling any remote call, then execute a small, resumable Qwen3-VL pilot, run the approved experimental matrix, and package metrics and protocol evidence for the paper revision.

## Phases

- [ ] **Phase 1: Audit and visual manifests** — validate the supplied annotations and bind them to real observation frames.
- [ ] **Phase 2: Guarded Qwen3-VL pilot** — verify one-request transport, response parsing, recovery, and metrics within the approval cap.
- [ ] **Phase 3: Reviewer experiment matrix** — run the approved Base/Novel, horizon, image-count, action-pool, and seed comparisons.
- [ ] **Phase 4: Analysis package** — summarize results, failure modes, cost metadata, and paper-ready evidence.

## Phase Details

### Phase 1: Audit and visual manifests

**Goal:** Produce a data audit and valid, reproducible pilot manifests without invoking any API.

**Depends on:** Nothing.

**Requirements:** DATA-01, DATA-02, DATA-03.

**Success Criteria:**
1. The audit reproduces the paper's Table II split counts and reports any observed discrepancy.
2. The audit identifies all action-pool and annotation integrity issues without altering source data.
3. A manifest builder rejects missing, unreadable, ambiguous, or incorrectly sized visual observations.
4. Once source frames are supplied, a 10-Base plus 10-Novel pilot manifest validates every image and action mapping.

**Plans:** TBD.

### Phase 2: Guarded Qwen3-VL pilot

**Goal:** Complete approved low-volume Qwen3-VL requests with durable provenance and exact output validation.

**Depends on:** Phase 1.

**Requirements:** SAFE-01, SAFE-02, API-01, EVAL-01, EVAL-02.

**Success Criteria:**
1. No API call is possible under the checked-in configuration.
2. A reviewed pilot saves raw responses, parsed predictions, non-secret request metadata, usage, latency, and retry history.
3. Restarting a run skips successful samples and never silently changes a prediction.
4. Metrics include all valid manifest samples, including failures.

**Plans:** TBD.

### Phase 3: Reviewer experiment matrix

**Goal:** Run the approved matrix under both original and Total-pool action protocols.

**Depends on:** Phase 2.

**Requirements:** EXP-01.

**Success Criteria:**
1. Results identify every split, horizon, image setting, pool type, model ID, prompt version, and seed.
2. Original protocol results use Base→Base and Novel→Novel action pools.
3. Unified protocol results use the Total pool for both splits.
4. Seed means and standard deviations are reported without pooling incomparable configurations.

**Plans:** TBD.

### Phase 4: Analysis package

**Goal:** Produce revision-ready tables and failure analysis supported by saved runs.

**Depends on:** Phase 3.

**Requirements:** EVAL-02.

**Success Criteria:**
1. The report includes overall and macro-by-event metrics, API/parse failure rates, token usage, and latency.
2. Comparison tables state protocol differences and limitations of zero-shot proprietary-model evaluation.
3. Every reported aggregate links to immutable run metadata and predictions.

**Plans:** TBD.

## Progress

| Phase | Plans Complete | Status | Completed |
|---|---:|---|---|
| 1. Audit and visual manifests | 0/TBD | In progress | — |
| 2. Guarded Qwen3-VL pilot | 0/TBD | Not started | — |
| 3. Reviewer experiment matrix | 0/TBD | Not started | — |
| 4. Analysis package | 0/TBD | Not started | — |

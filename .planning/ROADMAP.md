# Roadmap: OEPP Reviewer Revision

**Updated:** 2026-07-30
**Scope:** reviewer-driven evidence expansion. The canonical execution contract is `.planning/REVIEWER-REVISION-PLAN.md`.

## Goal

Address the three experimentally actionable reviewer gaps without changing the OEPP task definition during evaluation:

1. add a current Qwen3.5 hosted MLLM Table V baseline;
2. establish robustness under a recovered, independently frozen alternate event split;
3. assess and, where technically valid, evaluate the public P3IV and KEPP procedure-planning methods.

## Phases

### Phase 1: Qwen3.5 Table V baseline — Priority S

**Goal:** Evaluate `Qwen/Qwen3.5-397B-A17B` through SiliconFlow using the audited historical Table V `T=4`, 3+3-image comparator protocol.

**Fixed contract:** Base 855 and Novel 1,297 observations; source-order split-specific pools; split-specific legacy v2 prompts; `temperature=0`; `enable_thinking=false`; no retry; `paper_compatible` main metrics plus `strict` audit.

**Acceptance criteria:**

1. A 2 Base + 2 Novel pilot demonstrates exact returned model, six-image transport, four-action parser success, `finish_reason=stop`, zero API failures, and zero reasoning tokens.
2. Each full split has one-to-one manifest/request/response/prediction coverage; failures remain in the denominator.
3. Results retain request configuration, usage, latency, raw responses, model identity, and both scoring modes.
4. The report names SiliconFlow, date, model ID, prompt, candidate order, image preprocessing and provider limitations; it does not make a controlled capability claim.

**Status:** complete and audited. Base has 855 / 855 API success; Novel has 1,294 / 1,297 API success, with three retained retryable 503 failures. Full audit: `oepp_api_eval/runs/siliconflow_qwen35_397b_full_audit_20260730.json`.

### Phase 2: P3IV and KEPP feasibility and OEPP integration — Priority A

**Goal:** Determine whether each public method can be evaluated under OEPP without changing its benchmark task, then implement individually valid baselines.

**Acceptance criteria:**

1. Record upstream commit/license, dependencies, data representation, training target, inference horizon, candidate-action decoding, and evaluation protocol for P3IV and KEPP separately.
2. Map OEPP inputs and split-specific action pools to each method; reject any mapping that leaks Novel test labels or trains/selects on test data.
3. Establish fresh initialisation, validation-only checkpoint selection, and Base/Novel export/evaluation evidence for every portable method.
4. If a method is not portable, record the precise incompatibility and do not replace it with an unlabelled approximation.

**Status:** static feasibility audit complete at `.planning/P3IV-KEPP-FEASIBILITY.md`; server training remains blocked until a portable, validation-safe mapping and its required feature provenance are established.

### Phase 3: Alternate split robustness matrix — Priority B

**Goal:** Recover and validate the original alternate event split, then rerun one frozen baseline matrix on it.

**Acceptance criteria:**

1. Recover the alternate split from authoritative provenance, preserve the source files unchanged, and normalize only into a versioned OEPP split manifest matching the current JSON schema.
2. Audit event membership, action-pool membership, duplicate videos, counts, transferability rules, feature/frame availability, and Base/Novel leakage before any training.
3. Rerun the frozen baseline set — MLP, Transformer, PDPP, plus portable P3IV and KEPP — with identical feature, seed, epoch, and validation-selection rules per model.
4. Report split-specific metrics and uncertainty separately; never average incomparable split definitions or omit a failed method.

**Status:** blocked while the training server and the alternate split provenance are unavailable. Begin recovery locally when its source path or archive is found.

## Dependency order

```text
Qwen3.5 pilot → Qwen3.5 full audit
P3IV / KEPP feasibility → freeze baseline matrix
alternate split recovery + audit → rerun frozen matrix
```

## Out of scope for these phases

- Treating Experiment 4 embedding metrics as MLLM Table V metrics.
- Silently changing action pools, prompts, horizons, candidate order, image preprocessing, or scoring to make a new method fit.
- Full hosted API calls before a passing pilot and a private approval artifact.
- Claiming alternative-split robustness before the same frozen baseline set has been run.

# OEPP Qwen3-VL API Evaluation

## What This Is

A standalone, remote-API evaluation workspace for adding a Qwen3-VL baseline to the Open-Event Procedure Planning (OEPP) paper. It reads the existing OEPP annotations and action pools without modifying the training repository, then records reproducible visual API requests, raw responses, and paper-compatible metrics.

## Core Value

Produce auditable Qwen3-VL results whose inputs, candidate-action protocol, and failures can be independently reproduced without retraining a local model.

## Requirements

### Validated

- ✓ OEPP contains base-to-base and base-to-novel evaluation under supplied action spaces — `OEPP-TIP.pdf`, Sections III–V.
- ✓ The local annotations match the paper's Table II split counts: 1,285 train, 337 validation, 416 base-test, and 733 novel-test videos.
- ✓ The existing evaluator computes sequence success, per-step accuracy, and set IoU — `eval.py:260-315`.

### Active

- [ ] Build a read-only audited manifest pipeline for the local OEPP annotations.
- [ ] Add a gated, resumable Qwen3-VL remote API evaluator.
- [ ] Evaluate Base and Novel splits under split-specific and Total action-pool protocols.
- [ ] Report paper-compatible metrics, failure rates, request cost metadata, and candidate-order variation.

### Out of Scope

- Local Qwen3-VL weights, GPU inference, model training, or feature extraction — remote API evaluation only.
- Any modification of the existing OEPP training, model, data, checkpoint, or evaluation files — preserve the baseline repository.
- Full-scale API calls before a human reviews the data audit, visual assets, prompt, and pilot output — controls cost and protocol errors.

## Context

`OEPP-TIP.pdf` reports GPT-4o results for $T=3,4$, one or three frames per observation, Base and Novel splits, and the SR/Acc/mIoU metrics. Reviewer-driven Qwen3-VL experiments must retain the supplied action-space protocol and make the alternative Total-pool protocol explicit. The repository holds annotations, action pools, and precomputed features; it does not contain raw test video frames.

## Constraints

- Work only within this checkout; the original repository is data-source read-only for this experiment.
- API keys stay in environment variables or an ignored `.env`; never code, logs, manifests, or Git.
- Phase 1 makes zero API calls. The initial configuration has `api_enabled: false` and zero permitted calls.
- A valid visual manifest requires real, decodable frame files. Annotation timestamps or `.npy` features are not image inputs.
- Preserve action strings exactly. Primary candidate IDs follow the source action-pool file order; sensitivity runs use explicitly recorded seeded permutations.
- Store raw API responses and parsing failures. Invalid predictions remain failures in the metric denominator.

## Key Decisions

- **Independent workspace:** add `oepp_api_eval/` rather than changing OEPP source code. This makes the experiment portable and protects the paper baseline.
- **Observation index:** map annotated sequence windows to extracted frames through a separate JSONL index. The current annotations have timestamps only, so deriving frame paths would be an unverified assumption.
- **Data-URL image transport:** encode local, validated frame files per request; Qwen3-VL's documented OpenAI-compatible API accepts `image_url` data URLs. URLs and credentials are not persisted.
- **Gated execution:** no call path is enabled initially; resumption skips only successfully completed `sample_id`s and records every actual API attempt in a workspace ledger.
- **Protocol matrix:** distinguish the original split-specific action pools from the reviewer-facing Total-pool protocol; never mix their results.
- **Candidate order:** use source action-pool file order for the paper-comparable primary result. This checkout contains no GPT-4o evaluator or prompt, so its original action-list order is unverified; report seeded candidate-order permutations as separate sensitivity results, not as a replacement primary protocol.

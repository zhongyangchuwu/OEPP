# P3IV and KEPP OEPP Feasibility

**Updated:** 2026-07-30
**Consumer:** reviewer-revision Phase 2 implementation decision.

## Evidence source

- KEPP official repository: `Ravindu-Yasas-Nagasinghe/KEPP`, public default branch inspected 2026-07-30.
- P3IV official repository: `SamsungLabs/procedure-planning`, public default branch inspected 2026-07-30.
- Current OEPP contracts: `dataset/dataset.py`, `embedding_preflight.py`, `feature_paths.py`, and the validated split-1 VideoCLIP protocol.

This is a static source audit, not a successful OEPP run.

## KEPP — conditionally portable

| Aspect | Upstream implementation | OEPP implication |
|---|---|---|
| Data | CrossTask, COIN, NIV | OEPP is built from CrossTask/COIN, so dataset lineage aligns. |
| Visual input | Per-video `.npz` with `frames_features`; loader concatenates 3 neighbouring frame features | OEPP S3D records use `frames_features`; the current final results use VideoCLIP `[frames, 768]` and cannot be substituted without a dimension-aware KEPP adapter. |
| Supervision | Step model, then procedural knowledge graph (PKG), then planning model | PKG and all conditioning paths must be constructed from OEPP training records only; validation selects checkpoints; Base/Novel must never enter graph construction or selection. |
| Labels | Dataset-specific action coding and one-hot actions | Need a versioned mapping from each OEPP split pool to the model class space, with unknown test actions rejected rather than remapped. |
| Runtime | Python/CUDA, multi-stage scripts and hard-coded arguments | Requires a reproducible adapter, run metadata, and staged validation checks before it can be called an OEPP baseline. |

**Decision:** attempt a KEPP OEPP adapter first. It is not a drop-in run: feature protocol, label mapping, and train-only PKG construction are mandatory gates.

## P3IV — material adaptation required

| Aspect | Upstream implementation | OEPP implication |
|---|---|---|
| Data | CrossTask, COIN, NIV; released CrossTask split checkpoint | OEPP must not reuse upstream closed-set `datasplit.pth`; use only a newly frozen OEPP Base/Novel split. |
| Visual state | Hard-coded `512` S3D + `128` VGGish audio input | Current validated OEPP final protocol is 768-D VideoCLIP. A VideoCLIP-only port changes the published input architecture; the alternative needs the original S3D and audio provenance. |
| Action head | CrossTask class size hard-coded to 106 | OEPP pools are Base 122, Novel 55, Total 161; class-head and decoding changes are required. |
| Inference | Probabilistic sampling and Viterbi over a training-derived transition matrix | Freeze sampling seed/count and build transitions from training only; report decoder choice separately. |
| Runtime | Python 3.7/CUDA-era code with paths and dimensions embedded in entry scripts | Needs a maintained compatibility environment or a minimal, tested port; an unlabelled rewrite is not acceptable as P3IV. |

**Decision:** retain P3IV as a requested target, but do not label any result as an unmodified P3IV baseline. The first implementation milestone is a written adapter design choosing either (a) original S3D+audio provenance or (b) an explicitly named VideoCLIP-only adaptation, followed by a smoke test. This decision cannot be made from model outputs.

## Shared gates before training

1. Obtain training-server access and verify the exact feature archives required by the selected method.
2. Freeze OEPP train / validation / Base / Novel JSON inputs and their action-ID mapping; hash every source file.
3. Add a conversion/adapter that validates dimensions, action ranges, split membership, and no test-data reads during training or PKG/transition construction.
4. Run a small training smoke test from random initialisation; confirm validation-only checkpoint selection.
5. Run Base and Novel independently and preserve predictions, metrics, seeds, decoder settings, and failures.

## Current blocker

The training server and alternate-split provenance are unavailable. KEPP/P3IV can be designed and source-audited locally, but no valid OEPP training/inference result can be claimed until feature provenance, runtime environment, and the above gates are exercised.

# Open-Event Procedure Planning

OEPP is a versioned Python package for Open-Event Procedure Planning baselines, embedding experiments, and auditable hosted-MLLM evaluations.

High-level reviewer records, paper material, result summaries, and server operations live in the unversioned sibling `../OEPP-project/`. Historical Table V protocol evidence is retained read-only at `../archive/legacy-gpt4o-evaluation/`.

## Layout

```text
src/oepp/
  data/          split bundles, feature paths, windows, PyTorch datasets
  models/        direct MLP/Transformer and PDPP model components
  training/      fresh direct/PDPP runners and checkpoints
  exports/       validation-selected embedding exporters
  evaluation/    metrics and figures
  api/           manifests, Table V protocol, parsing, runner, scoring
configs/
  training/      direct and legacy baseline configurations
  api/           checked-in offline MLLM configurations and prompts
data/
  splits/        versioned split manifests and source hashes
tests/           one package-level regression suite
```

## Environment

Python 3.10+ and `uv` are required. The root `pyproject.toml` and `uv.lock` are the only environment definition.

```bash
# CPU inspection and tests
uv sync --locked --extra cpu

# CUDA 11.8 training server
uv sync --locked --extra cu118
```

The package installs an `oepp` command. Every command below runs from the repository root.

## Versioned data contracts

`data/splits/split-001/manifest.json` freezes the canonical OEPP split-1 source paths and SHA-256 hashes. It keeps the original JSON files in place without duplicating them, while exposing named partitions and pools instead of numeric `is_val` values.

```bash
uv run oepp split-audit --data-root data --split-id split-001
```

A future shuffled event split must create a new bundle with independent provenance, record hashes, action pools, and leakage audit. It must not overwrite `split-001` or reuse its test records during train/validation construction.

### Authoritative alternate event-split import

`oepp import-alternate-split` accepts an externally authored membership JSON; it never creates a split or shuffles records. The input must declare `oepp-alternate-split-membership-v1`, the exact source split ID and ten source hashes, authority/source/event-rule provenance, and a complete, unique `(dataset, vid)` assignment for all four partitions. It derives the action pools in the frozen total-pool order and writes an immutable `data/splits/<alternate-id>/` bundle only when all gates pass.

```bash
uv run oepp import-alternate-split \
  --data-root data \
  --source-split-id split-001 \
  --membership private/authoritative-alternate-membership.json \
  --videoclip-root "$OEPP_VIDEOCLIP_ROOT" \
  --report runs/split-audits/split-002.json
```

The import rejects duplicate or incomplete memberships, source-hash mismatch, Base/Novel event leakage, missing or malformed VideoCLIP `[frames, 768]` arrays, and any attempt to overwrite a bundle or report. Action overlap is reported rather than rejected: canonical Base/Novel pools already share action strings. It verifies the generated bundle through `SplitBundle.load`; it does not train a model or turn an unverified source into a result.

VideoCLIP arrays are not committed. Place them under `features/OEPP_videoclip/`, or set `OEPP_VIDEOCLIP_ROOT` to an external directory. The package checks every requested feature path and its feature dimension before use.

## Fresh embedding baselines

Direct MLP and Transformer training use fresh initialization, validation-only checkpoint selection, deterministic train-sampling seed, and run metadata containing the selected split hashes.

```bash
uv run oepp train \
  --config configs/training/transformer.yaml \
  --data-root data \
  --run-dir runs/training/transformer/split-001-seed42

uv run oepp export \
  --checkpoint runs/training/transformer/split-001-seed42/best.pt \
  --data-root data \
  --output-dir runs/exports/transformer/split-001-seed42
```

PDPP is a separate diffusion runner and exporter. It uses the same named split bundle and keeps the validation checkpoint boundary intact.

```bash
uv run oepp train-pdpp --data-root data --split-id split-001 --horizon 3 --feat videoclip
uv run oepp export-pdpp --checkpoint path/to/best.pt --data-root data
```

The historical discrete-metric scripts remain explicitly isolated under `oepp.legacy`; they are not used by the fresh embedding runners.

## KEPP compatibility preflight

`kepp-preflight` is a local-only gate, not an implementation or training command. It constructs a deterministic PKG from `train` annotations only, audits action-class coverage per partition, and reports protocol blockers without executing or vendoring upstream KEPP code.

```bash
uv run oepp kepp-preflight --data-root data --split-id split-001 --feature videoclip \
  --output runs/preflight/kepp-split-001.json
```

Exit code `2` means the adapter is blocked; treat its JSON report as an approval input, not a failed training run. A full report is written only with `--output`; console output is bounded to the compatibility summary.


### P3IV compatibility preflight

`oepp p3iv-preflight` audits the frozen OEPP contract against the checked-out upstream P3IV surface without training, executing, copying, or adapting P3IV. It records the separate COIN and CrossTask input/class requirements, upstream split/decoder assumptions, source completeness, current feature mismatch, and train-only Novel action gap.

```bash
uv run oepp p3iv-preflight --data-root data --split-id split-001 --feature videoclip \
  --upstream-root ../upstreams/procedure-planning \
  --output runs/preflight/p3iv-split-001.json
```

Exit code `2` is expected while the unmodified method is blocked. A later VideoCLIP-only implementation requires an explicit adaptation decision and must not be described as upstream P3IV.

## Hosted-MLLM evaluation

The API evaluator is `oepp.api`, not a nested project. It retains the audited Table V legacy adapter, including prompt version, candidate order, parser, failure preservation, and separate `paper_compatible` / `strict` scores.

Checked-in API configurations are offline:

```yaml
api_enabled: false
max_calls: 0
```

Copy `configs/api/.env.example` to the ignored root `.env`; keep credentials only in environment variables or that ignored file. Any run above ten calls requires a private approval artifact. Never enable a public configuration in place.

```bash
uv run oepp api-build-manifest --help
uv run oepp api-build-tablev --help
uv run oepp api-run --help
uv run oepp api-evaluate --help
```

### Offline Table V preparation

The audited legacy adapter supports only `T=3` and `T=4`; its 3+3 image contract uses `start_f+[0,1,2]` and `end_f+[-2,-1,0]`. Parameterizing `T` does not authorize a provider call.

```bash
uv run oepp api-extract-tablev --horizon 3 --help
uv run oepp api-build-tablev --horizon 3 --help
uv run oepp api-replay-tablev --horizon 3 --help
```

`api-replay-tablev` converts legacy action-list artifacts into the current frozen split/window contract and reports `paper_compatible` and `strict` separately. It never transmits images or calls an API. Coverage remains explicit: historical rows absent from a source-window set are reported, not invented or silently scored.

### Shared frame-cache preparation

`api-build-video-index` resolves a read-only `(dataset, vid) → source_video_path` index against frozen annotation windows. `api-plan-frame-cache` then creates one isolated plan for the complete `T=3/T=4 × 1+1/3+3` grid without opening a video. The versioned `1+1` frame contract is `start_f+[0]` and `end_f+[0]`; `3+3` retains the legacy offsets above.

```bash
uv run oepp api-build-video-index --help
uv run oepp api-plan-frame-cache --help
uv run oepp api-extract-frame-cache --help
uv run oepp api-compose-frame-cache --help
uv run oepp api-verify-frame-cache-parity --help
```

Use a new cache ID and ignored paths (`private/` for the source index, `runs/` for plans and indexes, `sampled_frames/` for JPEGs). The extractor batches requests by source video and records every unavailable frame; the composer preserves any dependent unavailable observation. Run the composer and `api-build-tablev --frame-protocol shared-cache` once for each horizon, image setting, and Base/Novel split. Historical `T=4` frames, manifests, and runs are immutable.

Before a server-wide extraction, use `api-verify-frame-cache-parity` to compare the new `T=4`, 3+3 JPEG hashes, timestamp records, image order, and frozen-window identities. If the parity gate detects decoder drift, seed a fresh cache with the immutable `T=4`, 3+3 observations and frame root through `api-extract-frame-cache --legacy-observations ... --legacy-frame-root ...`. Conflicting JPEG bytes for one `(source video, timestamp)` are rejected, and no command initiates an API request.


API requests, responses, manifests, sampled frames, logs, checkpoints, and exports are ignored under `runs/` or other ignored artifact paths. They are never committed.

## Tests and quality checks

```bash
uv run python -m unittest discover -s tests -t . -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests cover split-bundle hashes and windows, feature resolution, checkpoint selection, export alignment, PDPP single-GPU behavior, candidate ordering, API safety, parser behavior, Table V frame protocol, and dual scoring.
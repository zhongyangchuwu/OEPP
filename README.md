# Open-Event Procedure Planning

OEPP is a versioned Python package for Open-Event Procedure Planning baselines, embedding experiments, and auditable hosted-MLLM evaluations.

High-level reviewer records, paper material, result summaries, and server operations live in the unversioned sibling `../OEPP-project/`. Historical protocol evidence remains read-only in `../OEPP_server/`.

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

### Offline Table V T=3 preparation

The Table V adapter supports only the audited legacy horizons `T=3` and `T=4`; it always keeps three start and three goal images with offsets `start_f+[0,1,2]` and `end_f+[-2,-1,0]`. Parameterizing `T` does not authorize a provider call.

```bash
uv run oepp api-extract-tablev --horizon 3 --help
uv run oepp api-build-tablev --horizon 3 --help
uv run oepp api-replay-tablev --horizon 3 --help
```

`api-replay-tablev` converts legacy action-list artifacts into the current frozen split/window contract and reports `paper_compatible` and `strict` separately. It never transmits images or calls an API. Coverage remains explicit: historical rows absent from a source-window set are reported, not invented or silently scored.


API requests, responses, manifests, sampled frames, logs, checkpoints, and exports are ignored under `runs/` or other ignored artifact paths. They are never committed.

## Tests and quality checks

```bash
uv run python -m unittest discover -s tests -t . -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests cover split-bundle hashes and windows, feature resolution, checkpoint selection, export alignment, PDPP single-GPU behavior, candidate ordering, API safety, parser behavior, Table V frame protocol, and dual scoring.
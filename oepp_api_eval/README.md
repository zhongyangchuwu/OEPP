# OEPP Qwen3-VL API Evaluation

Independent, remote-API-only evaluation for the OEPP paper. It reads `../data/` but never modifies OEPP source code, annotations, checkpoints, features, or model outputs.

## Current state

`python src/inspect_data.py` reproduces the paper's Table II counts from the local annotations. The checkout has **no raw test videos or frame files**: `dataset/dataset.py` consumes precomputed `.npy` features from external absolute paths, while the JSON annotations only provide segment timestamps. Therefore, a visual manifest and every API call are intentionally blocked until a verified observation index maps each annotated sequence window to real local frames.

The checked-in Qwen configuration is Phase 1 safe:

```yaml
api_enabled: false
max_calls: 0
```

No command in this project can issue an API request in that state.

## Environment

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
cd oepp_api_eval
uv sync
uv run python src/inspect_data.py --output-dir data_audit
```

Create an ignored `.env` from `.env.example`; keep model ID and endpoint configurable because availability depends on the Model Studio account and region:

```dotenv
DASHSCOPE_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-235b-a22b-instruct
```

`DASHSCOPE_API_KEY` is never read into a config snapshot, a log, or a JSONL artifact.

## Table V T=4, 3+3 API protocol

The primary reviewer baseline uses the final GPT row's observation protocol: a four-action
window, three start frames at `start_f + [0, 1, 2]`, three goal frames at
`end_f + [-2, -1, 0]`, and split-specific action pools. The Table V source manifests and
raw MP4 videos live on the server, not in this repository. The checked-in Table V Qwen
configuration remains disabled and uses the legacy numbered-action response parser:

```yaml
experiment:
  api_enabled: false
  max_calls: 0
response_parser: legacy_numbered_action_names
```

### Server: extract portable observations

Run this from a checkout of this fork on the video server. The extractor reads the historical
sequence manifests, writes six verified JPEGs per available sequence under `sampled_frames/`,
and writes only relative image paths to the observation index. Missing videos are recorded,
not silently skipped.

```bash
cd oepp_api_eval
uv sync

uv run python src/extract_tablev_frames.py \
  --sequence-file /data1/wuyilu/OEPP/LLM/T=4_base.json \
  --split base \
  --frame-root sampled_frames/tablev_t4_3x3 \
  --observations-output manifests/tablev_t4_3x3/base_observations.jsonl \
  --unavailable-output manifests/tablev_t4_3x3/base_unavailable.jsonl

uv run python src/extract_tablev_frames.py \
  --sequence-file /data1/wuyilu/OEPP/LLM/T=4_novel.json \
  --split novel \
  --frame-root sampled_frames/tablev_t4_3x3 \
  --observations-output manifests/tablev_t4_3x3/novel_observations.jsonl \
  --unavailable-output manifests/tablev_t4_3x3/novel_unavailable.jsonl
```

For a pilot, append `--limit 5` to each command. Synchronize the whole
`sampled_frames/tablev_t4_3x3/` directory and both observation files to the local workspace;
keep their relative layout unchanged.

### Local: validate the manifest and run an approved API job

Build a local manifest only after the frames have arrived. The builder cross-checks every
sequence against the local OEPP annotation and verifies all six image files.

```bash
cd oepp_api_eval
uv run python src/build_tablev_manifest.py \
  --observations manifests/tablev_t4_3x3/base_observations.jsonl \
  --data-root ../data \
  --frame-root sampled_frames/tablev_t4_3x3 \
  --split base \
  --output manifests/tablev_t4_3x3/base_manifest.jsonl
```

Repeat with `--split novel` and `novel_observations.jsonl`. Create a private configuration
copy from `configs/qwen3vl_tablev_t4_3x3.yaml`; only that private copy may set
`api_enabled: true` and a reviewed call limit. It uses the exact provider model ID from
`QWEN_MODEL`, saves raw responses and usage, and leaves malformed action names as failures.

```bash
mkdir -p private
cp configs/qwen3vl_tablev_t4_3x3.yaml private/qwen3vl_tablev_t4_3x3.yaml
uv run python src/run_api.py \
  --config private/qwen3vl_tablev_t4_3x3.yaml \
  --manifest manifests/tablev_t4_3x3/base_manifest.jsonl \
  --run-name qwen-tablev-t4-3x3-base
```

Run `evaluate.py` and `summarize_results.py` separately for Base and Novel. Do not aggregate
the two runs until their protocol, model ID, prompt version, and candidate-order metadata match.

## Phase 1: audit and visual manifests

The audit reports files, fields, sample/segment/horizon counts, action-pool integrity, duplicate video identifiers, and frame availability:

```bash
uv run python src/inspect_data.py --data-root ../data --output-dir data_audit
```

Supply a JSONL observation index only after extracting frames from an identified source. One row binds one OEPP sequence window to actual local frame files:

```json
{
  "dataset": "COIN",
  "vid": "0R9pdc9dO3Q",
  "start_step": 0,
  "end_step": 2,
  "image_setting": "1+1",
  "start_images": ["/absolute/path/start.jpg"],
  "end_images": ["/absolute/path/goal.jpg"]
}
```

For `3+3`, each image list contains exactly three paths. The builder checks all files with Pillow, preserves original action text, assigns sequential IDs, and rejects any missing or ambiguous observation. The faithful primary protocol preserves the action-list file order; seeded shuffles are separate candidate-order sensitivity runs.

```bash
uv run python src/build_manifest.py \
  --data-root ../data \
  --observation-index manifests/observations.jsonl \
  --split base novel \
  --horizons 3 4 \
  --image-settings 1+1 3+3 \
  --pool-type split \
  --candidate-order original \
  --output manifests/oepp_split_pool.jsonl
```

For generic manifests, candidate-list order is not specified by the paper's GPT code; use `--candidate-order original` for the primary run, then report seeded permutations separately. The Table V $T=4$, 3+3 workflow above is a distinct, source-backed legacy protocol with its own prompt, parser, and split-specific pools. Never aggregate generic, legacy, Total-pool, or shuffled-candidate results together.

## Guarded API execution

After data/prompt review and explicit approval, enable API access in a private run configuration and run a maximum of the approved samples. `run_api.py`:

- refuses disabled configurations and missing environment values;
- maintains a workspace-wide attempt ledger;
- skips only completed `sample_id`s after restart;
- retries only transient HTTP/network failures with exponential backoff;
- keeps local images in memory as data URLs, never writes base64 or an API key;
- saves request metadata, raw responses, parsed predictions, failures, usage, latency, and retry count separately.

```bash
uv run python src/run_api.py \
  --config configs/qwen3vl.yaml \
  --manifest manifests/pilot.jsonl \
  --run-name pilot-qwen3vl \
  --max-calls 10
```

The runner requires a reviewed approval JSON for any limit over ten. The pilot configuration remains at zero by design; do not change it before review.

## Evaluation

```bash
uv run python src/evaluate.py \
  --manifest manifests/pilot.jsonl \
  --predictions runs/pilot-qwen3vl/predictions.jsonl \
  --output runs/pilot-qwen3vl/metrics.json
uv run python src/summarize_results.py \
  --metrics runs/pilot-qwen3vl/metrics.json \
  --output runs/pilot-qwen3vl/summary.md
```

Metrics match OEPP semantics: SR is exact sequence match, Acc is position-wise accuracy, and mIoU is set IoU. Missing, malformed, and out-of-pool model outputs stay in the denominator as zero-score failures; they are never padded, truncated, reordered, or semantically replaced.

## Sources

- `../OEPP-TIP.pdf`, Table II and Table V: benchmark scope and prior GPT image/horizon experiments.
- `../dataset/dataset.py`: original split selection, action-pool convention, and feature-only external frame paths.
- [Qwen3-VL OpenAI-compatible API example](https://github.com/QwenLM/Qwen3-VL): documented `OpenAI` client and `image_url` input.

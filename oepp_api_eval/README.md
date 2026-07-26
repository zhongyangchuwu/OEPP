# OEPP Qwen3-VL API Evaluation

Independent, remote-API-only evaluation for the OEPP paper. It reads `../data/` but never modifies OEPP source code, annotations, checkpoints, features, or model outputs.

## Current state

`python src/inspect_data.py` reproduces the paper's Table II counts from local annotations. Raw MP4 videos remain server-only, but the verified Table V 3+3 frame sets and observation indices are now local: 855 Base observations (33 missing-video windows excluded) and 1,297 Novel observations. Checked-in configurations remain offline until an approved private copy is used.

The checked-in Qwen configuration is Phase 1 safe:

```yaml
api_enabled: false
max_calls: 0
```

No command in this project can issue an API request in that state.

中文任务进展、服务器提帧结果和问题记录见 [REPORT.md](REPORT.md)。

## Environment

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). The repository-root `pyproject.toml` and `uv.lock` are the only environment definition; this directory is not a nested uv project.

From the repository root:

```bash
uv sync --locked
uv run python oepp_api_eval/src/inspect_data.py --data-root data --output-dir oepp_api_eval/data_audit
```

The commands below enter `oepp_api_eval/` for relative data paths. uv discovers the root project, uses its root `.venv/`, and does not create a nested environment.

Create an ignored `.env` from `.env.example`. The active SiliconFlow Qwen3-VL configuration reads the key only from the shell and stores endpoint/model selection separately:

```dotenv
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_QWEN_MODEL=Qwen/Qwen3-VL-32B-Instruct
```

Export `SILICONFLOW_API_KEY` in the same shell. The key is never read into a config snapshot, a log, or a JSONL artifact.

### OpenRouter Gemini 3.1 Flash Lite

The checked-in OpenRouter Base/Novel Table V configurations are also offline. They use `OPENROUTER_API_KEY`, `https://openrouter.ai/api/v1`, and the account-verified model ID `google/gemini-3.1-flash-lite`; copy them into ignored `private/` before enabling any calls. Query the authenticated `GET /api/v1/models` inventory and retain its model/pricing snapshot before a pilot or full run. OpenRouter may route a request across eligible upstream providers, so keep its returned routing metadata with the raw response and never silently replace a region-unavailable model.

## Table V T=4, 3+3 API protocol

The primary reviewer baseline uses the final GPT row's observation protocol: a four-action
window, three start frames at `start_f + [0, 1, 2]`, three goal frames at
`end_f + [-2, -1, 0]`, and split-specific action pools. Raw MP4 videos live on the server,
while local frame/observation artifacts are sufficient for API inference. The checked-in
SiliconFlow Table V configuration remains disabled and uses the legacy numbered-action response parser:

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
uv sync --locked

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

Build the Novel manifest with `--split novel` and `novel_observations.jsonl`. The public
SiliconFlow configurations lock each historical prompt variant to its intended split; copy the
matching configuration privately before enabling API calls:

```bash
mkdir -p private
cp configs/siliconflow_qwen3vl32_tablev_t4_3x3.yaml private/qwen3vl32_tablev_t4_3x3_base.yaml
cp configs/siliconflow_qwen3vl32_tablev_t4_3x3_novel.yaml private/qwen3vl32_tablev_t4_3x3_novel.yaml

uv run python src/run_api.py \
  --config private/qwen3vl32_tablev_t4_3x3_base.yaml \
  --manifest manifests/tablev_t4_3x3/base_manifest.jsonl \
  --run-name siliconflow-qwen3vl32-base
uv run python src/run_api.py \
  --config private/qwen3vl32_tablev_t4_3x3_novel.yaml \
  --manifest manifests/tablev_t4_3x3/novel_manifest.jsonl \
  --run-name siliconflow-qwen3vl32-novel
```

The runner records raw responses and action text, rejects a manifest for the wrong configured split,
and retains malformed responses as failures.

## Phase 1: audit and visual manifests

The audit reports files, fields, sample/segment/horizon counts, action-pool integrity, duplicate video identifiers, and frame availability. Its sequence-window counts follow the evaluation protocol: a sequence shorter than $T$ contributes one left-padded window, rather than being excluded. Therefore, the current annotations contain 888 Base and 1,297 Novel $T=4$ windows before the 33 Base windows whose source videos are unavailable are excluded.

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

For `3+3`, each image list contains exactly three paths. The builder checks all files with Pillow and preserves the source action-pool text and order, including whitespace-only aliases. Action-name matching during parsing/scoring is case- and whitespace-insensitive, matching the historical Table V evaluator; seeded shuffles are separate candidate-order sensitivity runs.

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

For the primary Table V comparison, use `paper_compatible`: formatted four-action responses are
normalized by case/whitespace and receive the same action-wise and set-IoU treatment as the historical
notebook, even when an action is outside the candidate pool. Keep `strict` as the complementary format
and candidate-compliance report. API failures, missing predictions, and structurally invalid responses
remain zero-score denominator entries in both modes.

```bash
uv run python src/evaluate.py \
  --manifest manifests/tablev_t4_3x3/base_manifest.jsonl \
  --predictions runs/siliconflow-qwen3vl32-base/predictions.jsonl \
  --scoring-mode paper_compatible \
  --output runs/siliconflow-qwen3vl32-base/metrics_paper_compatible.json
uv run python src/evaluate.py \
  --manifest manifests/tablev_t4_3x3/base_manifest.jsonl \
  --predictions runs/siliconflow-qwen3vl32-base/predictions.jsonl \
  --scoring-mode strict \
  --output runs/siliconflow-qwen3vl32-base/metrics_strict.json
uv run python src/summarize_results.py \
  --metrics runs/siliconflow-qwen3vl32-base/metrics_paper_compatible.json \
  --output runs/siliconflow-qwen3vl32-base/summary_paper_compatible.md
```

Repeat separately for Novel. Never aggregate Base and Novel, generic, legacy, distinct prompt versions,
or distinct scoring modes.

## Sources

- `../OEPP-TIP.pdf`, Table II and Table V: benchmark scope and prior GPT image/horizon experiments.
- `../dataset/dataset.py`: original split selection, action-pool convention, and feature-only external frame paths.
- [Qwen3-VL OpenAI-compatible API example](https://github.com/QwenLM/Qwen3-VL): documented `OpenAI` client and `image_url` input.

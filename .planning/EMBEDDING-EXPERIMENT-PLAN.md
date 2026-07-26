# Experiment 4 — Embedding Prediction Analysis Plan

## Decision and scope

Run three independent fresh-initialization baselines under the shared `videoclip`, split 1, `T=3`, `is_pad=1` data protocol: `attention` is the primary Transformer result, `MLP` is its direct non-attention comparator, and PDPP is a separate stochastic diffusion baseline. Do not aggregate their checkpoints, outputs, or figures.

All three deliver validation-selected checkpoints plus reproducible Base/Novel embedding exports and the same pre-specified aggregate plots. This is not a search for a visually appealing projection.

### No checkpoint reuse

This experiment starts from fresh initialization. Do not load historical Transformer, MLP, or PDPP checkpoints. The repository's only `checkpoint/whl/checkpoint.txt` explicitly records `No file, we will update later`; it is not a model artifact. The fresh run must create the validation-selected `best.pt` consumed by the exporter.

## Current audit

- Both `TransformerEncoder` and `MLP` return a list of `T` predicted 768-dimensional embeddings. `Seq_action` yields the corresponding GT action-text embedding tensor, action names, labels, video ID, and endpoint feature tensor. PDPP samples the same action-embedding space from its EMA diffusion model.
- New `train_embeddings.py` runs fresh MLP/Transformer training and saves explicit `last.pt` and validation-selected `best.pt`; `export_embeddings.py` reconstructs the direct model and exports continuous embeddings. PDPP now saves the same fresh-run lifecycle and `export_pdpp_embeddings.py` exports seeded diffusion samples. Legacy `train.py`, `eval.py`, and PDPP `--evaluate` are not pure embedding export paths.
- `Seq_action` now stores stable source-window metadata without changing its returned batch tuple, loads its action pool once, and rejects an unknown action instead of silently mapping it to label zero.
- No valid local checkpoint exists. The local environment has neither Torch nor a CUDA device; `/data0` and `/data1` are not mounted. No SSH experiment host is configured in this harness. The experiment must not start until a server host/access path is supplied and preflight succeeds.
- For `T=3`, `is_pad=1`, the current annotations generate 3,550 train, 883 validation, 1,138 Base-test, and 1,691 Novel-test windows. Base and Novel exports contain 2,829 windows / 8,487 step embeddings. Saving float32 predicted and GT tensors requires about 49.73 MiB before metadata and plots.

## Execution evidence — 2026-07-26

- Server preflight passed with CUDA-enabled PyTorch 2.3.0+cu118 and all 2,771 VideoCLIP arrays present, valid, and matched to the annotations.
- Fresh 200-epoch direct runs completed under `results/experiment4/attention_t3_seed42_run1/` and `results/experiment4/mlp_t3_seed42_run1/`. Validation selected Transformer epoch 152 (`SR=25.82%`, `Acc=55.61%`, `mIoU=59.48%`) and MLP epoch 145 (`SR=27.63%`, `Acc=56.21%`, `mIoU=61.14%`); neither selection used Base or Novel test data.
- Fresh 200-epoch PDPP completed under `results/experiment4/pdpp_checkpoints/pdpp_t3_seed42_run1/`; `best.pt` is epoch 168 (`SR=30.58%`, `Acc=56.93%`, `mIoU1=68.80%`) while `last.pt` is epoch 200. The export uses the selected EMA checkpoint, not `last.pt`.
- Final server-side exports exist at `embedding_results/{attention_t3_seed42_run1,mlp_t3_seed42_run1,pdpp_t3_seed42_run1}/`. Every model has Base `(1138, 3, 768)`, Novel `(1691, 3, 768)`, 3,414 / 5,073 step rows, summaries, and all seven pre-specified figures.
- Repeating the final PDPP export with `sampling_seed=42` produced bitwise-equal Base and Novel predicted/GT raw tensors. Local artifact download is intentionally deferred after WSL rsync stalled; server artifacts are canonical until a resumable archive transfer is prepared.

## Required implementation

### 1. Reproducible training and checkpointing

`train_embeddings.py` preserves the existing direct MLP/Transformer model, feature, candidate-pool, and legacy loss semantics while adding explicit state-dict `last.pt`/`best.pt` checkpoints. Direct models select by validation SR, then validation Acc, then lower validation MSE, then earlier epoch. PDPP checkpoints store its EMA state and select only by validation SR then Acc; its seeded exporter reports embedding MSE independently.

Each checkpoint records epoch, model/optimizer state, seed, complete config/arguments, validation metrics, action-embedding and annotation hashes, and code-run provenance. No Base or Novel test metric participates in selection. PDPP Base/Novel-per-epoch evaluation is disabled unless an explicit `--test_during_training` diagnostic flag is supplied.

### 2. Eval-only exporters

`export_embeddings.py` and `export_pdpp_embeddings.py` are separate from training. They reconstruct the matching fresh checkpoint, call `model.eval()`, use `torch.inference_mode()`, process Base and Novel independently with `shuffle=False`, and assert every exported action sequence matches `Seq_action` metadata. PDPP export uses a fixed sampling seed because diffusion sampling otherwise begins from random noise.

Each exporter saves raw float32 predicted/GT tensors, candidate embeddings once per run, every window-step metric row, split/step summaries, and the same named figures. They fail rather than silently substitute an absent checkpoint, unknown action, missing feature, or metadata mismatch.

### 3. Pre-specified figures

All figures are generated from the complete Base/Novel export, with error bars from a fixed-seed bootstrap over windows. No plot is selected after inspecting whether it supports a desired conclusion.

1. `cosine_by_step.png`: Base and Novel mean cosine(predicted, GT) for each step, 95% CIs.
2. `mse_by_step.png`: Base and Novel raw mean MSE for each step, 95% CIs; report the zero-vector MSE reference because embedding norms are not unit length.
3. `cosine_distribution_by_step.png`: full window-level cosine distribution by split and step.
4. `candidate_margin_by_step.png`: GT cosine margin versus the strongest non-GT candidate, with the zero reference line.
5. `base_vs_novel.png`: split-level cosine, MSE, top-1 accuracy, and GT-rank summary with CIs.
6. `correct_vs_wrong.png`: cosine and margin stratified by candidate top-1 correctness; label it descriptive, not causal.
7. `padding_sensitivity.png`: all windows versus non-padded windows, because Base has 80 and Novel has 147 padded `T=3` windows.

A deterministic PCA pair plot may be generated only as an appendix diagnostic. It must state the fitted population, seed, sampling rule, and explained variance. Do not use t-SNE/UMAP as primary evidence: their apparent cluster separation is not a metric of embedding prediction quality.

If a pre-specified result is counterintuitive, retain it and revise the conclusion. Omit only redundant views for page limits, then keep all outputs in the supplement/run directory. Do not cherry-pick favorable examples or projections.

## Validation

Before server training:

1. unit-test checkpoint tie-breaking, MSE/cosine/margin calculations, and exporter metadata alignment on a synthetic dataset;
2. run a two-window exporter dry run on the server and verify raw tensor shape, row count, hashes, and plot generation;
3. measure 10 training epochs under the final config and record seconds/epoch.

Server preflight is one command and must succeed before calibration:

```bash
uv run python embedding_preflight.py \
  --feature videoclip \
  --verify-feature-content \
  --output results/experiment4/preflight_videoclip.json
```

It reads `features/OEPP_videoclip` by default, or the directory named by `OEPP_VIDEOCLIP_ROOT`. It checks CUDA/Torch, all four annotation files, every expected `<feature-root>/<dataset>_<vid>.npy` feature path and 768-dimensional feature shape, split-pool membership, and action-text embedding coverage. It exits nonzero on missing or inconsistent data. Run `uv run python -m unittest discover -s tests -v` before the 10-epoch calibration.

Server acceptance checks after a fresh run:

1. direct runs contain `last.pt`, `best.pt`, `selection.json`, `training_metrics.jsonl`, `config.yaml`, and `run_metadata.json`;
2. Base raw tensor shape is `(1138, 3, 768)`, Novel is `(1691, 3, 768)`; per-step CSV row counts are 3,414 and 5,073;
3. `summary_metrics.json` and every pre-specified figure exist; the exporter aborts instead of dropping a missing/misaligned sample;
4. PDPP's saved `best.pt` has the `oepp-pdpp-embedding-v1` format and exporting it twice with the same `--sampling_seed` produces equal raw tensors.

## Runtime model

The direct MLP/Transformer runner uses 14 train batches per epoch (`ceil(3550/256)`) and, with its default batched validation, four validation batches (`ceil(883/256)`). For a 200-epoch direct run, measure 10 complete epochs and estimate:

```text
direct full training ≈ 20 × measured 10-epoch duration + final Base/Novel export
```

PDPP is materially more expensive: the documented 200-epoch command performs 200 diffusion training steps per epoch, or 40,000 diffusion steps before validation/export. Calibrate PDPP separately; never reuse the direct-model timing estimate.

The batched Base/Novel exporter covers 2,829 windows and should be much shorter than retraining; visualization is CPU-bound and should be timed after export. Exact duration remains blocked on unavailable server GPU, filesystem throughput, and installed Torch/CUDA versions.

## Execution order

1. Run static checks and the checkpoint-selection test locally; run the full test suite on the server after installing Torch.
2. Commit or record the exact dirty-worktree patch and hashes before server execution.
3. Configure server access; run preflight and a 10-epoch timing calibration for each model family intended for publication.
4. Run each selected baseline from fresh initialization in a separate directory; preserve `last.pt`, validation-selected `best.pt`, logs, and metadata.
5. Export Base and Novel embeddings from each model's `best.pt`; generate every pre-specified figure and summary without cross-model aggregation.
6. Inspect results only after all artifacts exist; then write the scientific interpretation separately.

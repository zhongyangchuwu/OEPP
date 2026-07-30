# OEPP Server Environment and Operating Requirements

**Observed:** 2026-07-30
**Host:** `OEPP` (`gpuserver-10`)
**Writable workspace:** `/data1/wuyilu/OEPP-hjr` only.

## Verified environment

| Item | Observed value |
|---|---|
| Remote user | `wuyilu` |
| Workspace ownership / mode | `wuyilu:wuyilu`, `775` |
| Root project manager | `/data1/wuyilu/OEPP-hjr/.tools/uv/uv` `0.11.32` |
| Python runtime | PyTorch `2.3.0+cu118`; CUDA available |
| GPUs | 8 × NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each |
| GPU state at observation | GPUs 0–7 each used 1 MiB, 11,010 MiB free, 0% utilization; no visible compute process |
| `/data1` capacity | 7.3 TiB total, 5.2 TiB used, 2.1 TiB available |

## Assets present in the writable workspace

| Path | Observation |
|---|---|
| `data/` | Current split-1 annotations, Base/Novel pools, total pool, VideoCLIP and S3D action embeddings. The directory listing contains no alternate-split manifest. |
| `features/OEPP_videoclip/` | 2,771 `.npy` VideoCLIP arrays; parent `features/` is about 1.3 GiB. |
| `OEPP_videoclip.zip` | Feature archive retained in the workspace. |
| `results/experiment4/` | Calibration and final Transformer / MLP / PDPP run directories, logs, checkpoints, and preflight reports; `results/` is about 6.1 GiB. |
| `embedding_results/` | Final, calibration, and repeat embedding exports; about 198 MiB. |
| `oepp_api_eval/` | API source, configs, manifests, sampled frames, tests, and an obsolete nested `.venv/`; no `runs/` directory exists on the server. |

## Repository state and preservation boundary

The remote checkout is on `experiment4-embedding-analysis` at `244199b`. Its own `.planning/STATE.md` is stale: it still says server access, feature deployment, and Qwen3.5 are unavailable. Treat the local checkout and its committed planning artifacts as the current project record.

The remote worktree has user-owned untracked material:

```text
embedding_results/
mise.toml
oepp_api_eval/tmp.py
results.tar
results/
```

These files and directories are evidence or local infrastructure. Do not run `git clean`, `git reset --hard`, broad deletion, or a blind merge/pull in the server workspace. Do not edit, move, or write any path outside `/data1/wuyilu/OEPP-hjr`.

## Required operating procedure

1. Work from the workspace root:

   ```bash
   cd /data1/wuyilu/OEPP-hjr
   export PATH="$PWD/.tools/uv:$PATH"
   ```

   Use the root `pyproject.toml` and `uv.lock`; do not activate or modify `oepp_api_eval/.venv/`.

2. Before changing server code, inspect the remote revision and untracked files. Synchronize only committed local source changes into the writable workspace by a reviewed operation. Preserve the listed result and infrastructure paths.

3. Before a GPU job, re-check occupancy. Select explicit devices with `CUDA_VISIBLE_DEVICES`; the all-idle state above is only an observation, not a reservation.

4. For the current OEPP learned baselines, use the root CUDA environment:

   ```bash
   .tools/uv/uv sync --locked --extra cu118
   .tools/uv/uv run python embedding_preflight.py \
     --feature videoclip \
     --verify-feature-content \
     --output results/experiment4/preflight_videoclip_current.json
   ```

   The preflight must pass before any fresh training or P3IV/KEPP adapter smoke test.

5. Place all generated checkpoints, logs, converted data, and exports under the workspace, using a dated, model-specific directory. Do not overwrite the completed `*_run1/` or calibration artifacts.

6. Keep API keys in ignored environment files only. API calls run locally unless a separate server-side private configuration, approval artifact, and explicit user instruction authorize them.

## Consequences for current reviewer tasks

- **Qwen3.5 API:** completed locally; no server rerun is required.
- **Alternate split:** not present in the inspected server `data/` listing. Recover authoritative provenance before creating any new manifest or baseline result.
- **KEPP / P3IV:** GPU capacity and current VideoCLIP assets are available, but neither method may train until its adapter design, feature provenance, action-ID mapping, and validation-only checkpoint protocol pass preflight.

## Evidence collection commands

Use read-only checks before every substantial run:

```bash
ssh OEPP 'cd /data1/wuyilu/OEPP-hjr && .tools/uv/uv --version'
ssh OEPP 'nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader'
ssh OEPP 'git -C /data1/wuyilu/OEPP-hjr status --short'
```

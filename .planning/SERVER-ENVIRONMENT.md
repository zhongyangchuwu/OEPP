# OEPP Server Environment and Operating Requirements

**Observed:** 2026-07-30
**Host:** `OEPP` (`gpuserver-10`)
**Writable workspace:** `/data1/wuyilu/OEPP-hjr` only.

## Repository roles

`OEPP/` in the local workspace is the sole authoring checkout: make, test, commit, and push every code or planning change there. `/data1/wuyilu/OEPP-hjr` is its runtime clone and execution workspace only. Never develop directly on the server or treat its stale Git revision as source of truth.

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

2. Deploy only an already committed local source revision. The server currently cannot complete `git ls-remote origin HEAD` over its HTTPS route (`GnuTLS recv error (-110)`), so do not rely on GitHub deployment until that tunnel path is retested successfully. Use the reviewed rsync procedure below in the interim.

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

## Deployment policy while server GitHub access fails

An integrity-checked 128 MiB incompressible rsync transfer through the `OEPP` tunnel completed in 34.358 seconds: **3.73 MiB/s** (31.25 Mbit/s), with matching local/remote SHA-256. This is suitable for code and moderate artifacts: approximately 200 MiB in 54 seconds and 1.3 GiB in 5.8 minutes at the observed rate. It is not a bandwidth reservation.

Use rsync only as a one-way local-to-server deployment. Never use `--delete`; the server contains results and user-owned untracked files. First inspect a dry run, then deploy the reviewed source tree:

```bash
LOCAL_ROOT=OEPP/
REMOTE_ROOT=OEPP:/data1/wuyilu/OEPP-hjr/
RSYNC_EXCLUDES=(
  --exclude=.git/ --exclude=.tools/ --exclude=.venv/ --exclude=.ruff_cache/
  --exclude=__pycache__/ --exclude=data/ --exclude=features/
  --exclude=results/ --exclude=embedding_results/ --exclude=OEPP_videoclip.zip
  --exclude=oepp_api_eval/.venv/ --exclude=oepp_api_eval/.env
  --exclude=oepp_api_eval/private/ --exclude=oepp_api_eval/runs/
  --exclude=oepp_api_eval/manifests/ --exclude=oepp_api_eval/sampled_frames/
)
sync_source() {
  rsync "$@" --itemize-changes --checksum "${RSYNC_EXCLUDES[@]}" \
    "$LOCAL_ROOT" "$REMOTE_ROOT"
}

# Inspect this output; do not deploy before reviewing it.
sync_source -an

# Deploy only after the dry-run itemization is accepted. No --delete.
sync_source -a --partial --mkpath
```

Record the local Git SHA and deployment timestamp under a new server result or deployment-metadata directory with each executed run. The server's `.git` revision remains the old clone commit after an rsync overlay; the deployment record, not `git rev-parse` on the server, identifies the runtime source version.

Use GitHub/Git deployment again only after the server can successfully run `git ls-remote origin HEAD` through the tunnel. A Git update remains preferable for source history when its transport is healthy; rsync is the safe operational fallback, not a replacement source of truth.

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

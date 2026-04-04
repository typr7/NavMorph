# Branch Status: `navmorph-multi-env-multi-gpu`

- Last updated: 2026-04-04
- Branch: `navmorph-multi-env-multi-gpu`
- HEAD: `ebc2194a469fee571e6d80caaf0a9d28aecbf5c5`
- Remote branch: `typr7/navmorph-multi-env-multi-gpu`

## Goal

Enable NavMorph train / eval to run with multiple environments and multiple GPUs, while removing the main single-environment assumptions that were breaking rollout and navigation.

## Current Status

- Multi-environment eval startup now passes with `GPU_NUMBERS=1` and `NUM_ENVIRONMENTS=4`.
- The earlier multi-environment crashes in rollout, graph feature construction, and navigation head indexing have been fixed on this branch.
- Final policy checkpoint loading is clean:
  - `model keys: 1674`
  - `checkpoint keys: 1674`
  - `missing: 0`
  - `unexpected: 0`
- This means the current eval checkpoint fully covers the current model definition.
- Full end-to-end eval metrics have not yet been reviewed in this status note.

## Key Branch Changes

### Main multi-env / multi-GPU support

- Added torchrun-compatible launcher path for train / eval / infer.
- Removed several single-device and single-environment assumptions from the NavMorph path.
- Kept per-environment rollout state batch-safe after env pause / filtering.

Relevant commits:

- `40dcd4f` — Enable multi-env multi-GPU train and eval

### Follow-up fixes from real runs

- `8a028c7` — Fix instruction token batching for eval
  - Fixed `batch_obs()` crash caused by Python lists without `.shape`.

- `2a3f11e` — Fix multi-env eval graph feature shapes
  - Fixed memory prompt broadcast leak and enforced per-env graph feature shapes before writing into `GraphMap`.

- `6bf42b6` — Fix multi-env navigation head indexing
  - Fixed hidden single-env indexing in `vilmodel_cmt.py` adaptive navigation head update.

- `ebc2194` — Log checkpoint coverage for strict-false loads
  - Added explicit coverage logs for auxiliary checkpoints and the final policy checkpoint.

## What Has Been Verified

- The branch can initialize multi-env eval with 4 environments on one process.
- Dataset / simulator / task initialization matches the requested multi-env setting.
- Policy setup completes successfully.
- The final eval checkpoint matches the current model exactly.

## What Still Needs Verification

- Full eval completion on `val_unseen`
- Final metrics sanity check
- Multi-GPU execution with `GPU_NUMBERS > 1`
- Agreement between single-env and multi-env eval outputs

## Known Caveats

- Auxiliary checkpoint loads are partial by design:
  - `cwp_predictor` only covers part of the full model
  - `NeRF_p16_8x8` also has many unexpected keys
- These auxiliary loads are overwritten by the final policy checkpoint during eval, so the final eval model is currently fully covered.
- Some code paths still depend on machine-specific pretrained checkpoint paths.
- Startup still emits old-environment warnings from Gym / torchvision / meshgrid APIs.

## Reference Commands

### Single-process multi-env eval

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
GPU_NUMBERS=1 \
NUM_ENVIRONMENTS=4 \
IL_BATCH_SIZE=4 \
SIMULATOR_GPU_IDS='[0]' \
TORCH_GPU_IDS='[0]' \
EVAL_CKPT_PATH=/data/data1/wzh/NavMorph/data/checkpoints/ckpt.pth \
MODEL_PRETRAINED_PATH=/data/data1/wzh/NavMorph/pretrained/model_step_100000.pt \
bash run_r2r/main.bash eval
```

### Multi-GPU entry rule

- `GPU_NUMBERS > 1` is required for `run_r2r/main.bash` to use `torchrun`.
- Setting only `CUDA_VISIBLE_DEVICES=...` does not enable multi-GPU execution by itself.

## Recommended Next Step

Run a full eval to completion and save the final metrics beside this status note.

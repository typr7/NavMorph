# Branch Status: `navmorph-multi-env-multi-gpu`

- Last updated: 2026-04-06
- Status: FAILED
- Branch: `navmorph-multi-env-multi-gpu`
- HEAD: `e61396fe3e25b4b7ae6f019ab10035cc0d22cfca`
- Remote branch: `typr7/navmorph-multi-env-multi-gpu`

## Goal

Enable NavMorph train / eval to run with multiple environments and multiple GPUs, while removing the main single-environment assumptions that were breaking rollout and navigation.

## Current Status

- This branch is marked as failed for its main objective.
- `GPU_NUMBERS=1` and `NUM_ENVIRONMENTS=1` training can run.
- The target line of work was multi-GPU / multi-env training close to `main` semantics.
- Repeated multi-GPU training tests still exposed new blocking failures:
  - stale unpublished checkpoint defaults
  - adaptive-head sync path errors
  - DDP reducer conflicts
  - segmentation checkpoint prefix mismatch
  - repeated NaN failures on the navigation path
- Several source-level numeric fixes were added, but this line has not reached a reliable multi-GPU training state.
- The branch should be treated as an unsuccessful exploration branch, not as a stable training branch.

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

- `ee326d6` — Improve training parity with main
  - Added rank sharding, global action-count normalization, and adaptive-head sync logic.

- `93ef889` — Remove stale default training checkpoint
  - Stopped defaulting train resume to a nonexistent `ckpt.iter25000.pth`.

- `25fc936` — Fix multi-GPU adaptive head sync
  - Corrected the adaptive-head lookup path under the ETP wrapper.

- `f89cb60` — Avoid DDP reducer conflicts in multi-GPU training
  - Replaced policy DDP usage with manual gradient synchronization.

- `764cf56` — Fix segmentation checkpoint key prefixes
  - Normalized `module.` prefixes when loading `segm.pt`.

- `b59b39e` — Fix NaNs in NeRF feature normalization
  - Clamped the NeRF feature norm to avoid zero-division NaNs.

- `e61396f` — Clamp viewpoint angle ratios to prevent NaNs
  - Clipped `arcsin` inputs in viewpoint-relative angle features.

## What Has Been Verified

- The branch can initialize multi-env eval with 4 environments on one process.
- Dataset / simulator / task initialization matches the requested multi-env setting.
- Policy setup completes successfully.
- The final eval checkpoint matches the current model exactly.
- Single-GPU training with `GPU_NUMBERS=1`, `NUM_ENVIRONMENTS=1`, `IL_BATCH_SIZE=1` can start.

## What Still Needs Verification

- Reliable multi-GPU training execution with `GPU_NUMBERS > 1`
- Agreement between single-GPU and multi-GPU training behavior
- Full end-to-end training stability after long runs
- Final eval completion and metrics sanity check

## Failure Decision

- Decision: stop this line as a failed branch.
- Reason:
  - the branch accumulated too many coupled fixes without reaching stable multi-GPU training
  - repeated testing kept surfacing new blockers after earlier blockers were fixed
  - the implementation no longer looks like a low-risk path to “multi-GPU training equivalent to `main`”
- Recommended follow-up:
  - keep this branch only as a record of attempted fixes
  - start a cleaner replacement branch from `main`
  - limit the next attempt to one narrower target, for example:
    - only single-env multi-GPU training
    - or only multi-env eval

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

Do not continue development on this branch. Start a fresh replacement branch from `main`.

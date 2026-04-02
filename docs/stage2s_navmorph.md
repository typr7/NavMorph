# Stage2S on NavMorph

## Goal
Stage2S rebuilds Idea 3 on top of NavMorph as a reachability-calibrated latent branch planner.
It has three jobs:
1. collect grouped counterfactual candidate sets from origin states
2. train isolated Stage2S heads offline
3. plug a calibrated planner back into NavMorph online evaluation

## Architecture in words
- **Host**: NavMorph still owns waypoint generation, panorama encoding, world-model latent updates, graph memory, and global navigation logits.
- **Stage2S logging**: `ss_trainer_ETP.py` probes grouped candidates, exports a structured latent state bundle, and writes grouped JSONL/GZip records.
- **Stage2S offline model**: `vlnce_baselines/stage2s/model.py` predicts one-step latent transition, executability, reachable progress, blocking risk, uncertainty, and semantic utility.
- **Stage2S planner**: `vlnce_baselines/stage2s/planner.py` shortlists semantically strong candidates, aggregates calibrated branch scores, and can override baseline top-1 when physics is poor.
- **Memory support**: `utils_p/memory.py` now accepts tagged surprise writes so physically informative events can enrich CEM instead of being silently ignored.

## Core files
- `run_r2r/stage2s_counterfactual.yaml` — grouped logging overlay
- `run_r2r/stage2s_navmorph.yaml` — online Stage2S eval overlay
- `scripts/stage2s/log_counterfactual_r2r.sh` — grouped counterfactual logging
- `scripts/stage2s/train_stage2s_offline.sh` — offline training wrapper
- `scripts/stage2s/eval_stage2s_navmorph.sh` — online eval wrapper
- `scripts/stage2s/summarize_stage2s.py` — offline/online summary and kill checks

## Commands
### 1. Grouped counterfactual logging
```bash
bash scripts/stage2s/log_counterfactual_r2r.sh
```
Key env vars:
- `SPLIT=val_seen|val_unseen`
- `EPISODE_COUNT=...`
- `OUTPUT_DIR=...`
- `PROBE_WIDTH=...`
- `CKPT_PATH=...`
- `PRETRAINED_PATH=...`

### 2. Offline Stage2S training
```bash
DEBUG_SYNTHETIC=1 bash scripts/stage2s/train_stage2s_offline.sh
# or real logs
TRAIN_RECORDS=data/logs/stage2s/val_seen.jsonl.gz \
VAL_RECORDS=data/logs/stage2s/val_unseen.jsonl.gz \
OUTPUT_DIR=data/logs/stage2s/offline_real \
bash scripts/stage2s/train_stage2s_offline.sh
```

### 3. Online Stage2S eval
```bash
bash scripts/stage2s/eval_stage2s_navmorph.sh \
  SPLIT=val_seen \
  EPISODE_COUNT=1 \
  TOP_K=2 \
  DEPTH=2 \
  CKPT_PATH=/path/to/navmorph_ckpt.pth \
  STAGE2S_CKPT=/path/to/stage2s_checkpoint.pt
```

## Phase gates
### Gate A — data protocol
- grouped log rows preserve `candidate_set_id`
- each origin state logs at least 3 probed candidates when available
- latent state bundle and candidate semantic bundle are present

### Gate B — offline learning
- executability AUROC beats the weak Stage2_v2 regime
- progress no longer collapses into raw distance
- ranking beats semantic-only top-1 on held-out grouped sets

### Gate C — online intervention
- changed rate must leave the negligible regime
- planner should produce selected/baseline branch diagnostics
- changed actions must be attributable to physical heads, not just wider search

### Gate D — paper bar
- matched-budget NavMorph baseline must be beaten credibly
- gains must survive 3 seeds
- final narrative must still stand against Stage 1 v2

## Failure signatures
- **Changed rate near zero**
  - likely causes: planner not wired, checkpoint missing, semantic shortlist too narrow, branch score dominated by semantic base score
- **Exec AUROC weak / ECE poor**
  - likely causes: bad candidate-state tensorization, latent mismatch, stale checkpoint, calibration absent
- **Progress strongly tracks raw distance**
  - likely causes: candidate-local representation too weak, semantic-only shortcut, bad target scaling
- **Planner only helps when search width increases**
  - likely causes: no real physical discrimination, branch scorer too close to semantic reranking
- **Selected branch often disagrees with real outcome**
  - likely causes: planner-real mismatch, missing surprise writes, host state bundle not aligned with runtime state

## Required ablations
- NavMorph baseline
- semantic-only shortlist control
- one-step vs depth-2 planner
- no calibration vs temperature calibration
- no blocking head
- no memory surprise writes
- comparison pointer to Stage 1 v2 main table

## Kill checks to watch in summaries
- changed rate still negligible
- exec calibration poor
- progress overly correlated with raw distance
- gains only appear when widening search

## Experiment ladder
1. synthetic trainer smoke
2. tiny grouped logging run on `val_seen`
3. tiny offline real-log debug run
4. one-episode online dry eval
5. short matched-budget eval
6. 3-seed main comparison only after the earlier gates hold

## Notes for future me
- Do not treat the current planner hook as paper-ready until the one-episode online dry eval is actually run in the full Habitat/NavMorph runtime.
- If online eval still changes almost nothing, stop and debug before increasing depth or width.

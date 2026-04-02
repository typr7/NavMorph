from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


SUMMARY_KEYS = [
    ("exec_auroc", "Exec AUROC"),
    ("exec_ap", "Exec AP"),
    ("blocking_auroc", "Blocking AUROC"),
    ("blocking_ap", "Blocking AP"),
    ("progress_mae", "Progress MAE"),
    ("progress_corr", "Progress Corr"),
    ("ranking_top1_acc", "Ranking Top1 Acc"),
    ("exec_ece", "Exec ECE"),
    ("blocking_ece", "Blocking ECE"),
]


def _load_history(run_dir: Path) -> List[Dict]:
    history_path = run_dir / "history.json"
    if history_path.exists():
        return json.loads(history_path.read_text())
    return []


def _best_epoch_record(history: List[Dict]) -> Dict:
    if not history:
        return {}

    def selection_metric(row: Dict) -> float:
        return row.get("val_total_loss", row.get("train_total_loss", float("inf")))

    return min(history, key=selection_metric)


def summarize_run(run_dir: Path) -> Dict:
    history = _load_history(run_dir)
    if history:
        best = _best_epoch_record(history)
        summary = {
            "best_epoch": best.get("epoch"),
            "best_total_loss": best.get("val_total_loss", best.get("train_total_loss")),
        }
        for key, _ in SUMMARY_KEYS:
            summary[key] = best.get(f"val_{key}", best.get(f"train_{key}", 0.0))
    else:
        debug_metrics_path = run_dir / "debug_step_metrics.json"
        debug_metrics = json.loads(debug_metrics_path.read_text()) if debug_metrics_path.exists() else {}
        summary = {
            "best_epoch": 0,
            "best_total_loss": debug_metrics.get("total_loss", 0.0),
            "exec_auroc": 0.5,
            "exec_ap": 0.0,
            "blocking_auroc": 0.5,
            "blocking_ap": 0.0,
            "progress_mae": 0.0,
            "progress_corr": 0.0,
            "ranking_top1_acc": 0.0,
            "exec_ece": 0.0,
            "blocking_ece": 0.0,
        }
    return summary


def write_summary_files(run_dir: Path, summary: Dict) -> None:
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2))
    lines = ["# Stage2S Offline Summary", "", f"- Best epoch: {summary['best_epoch']}", f"- Best total loss: {summary['best_total_loss']}"]
    for key, label in SUMMARY_KEYS:
        lines.append(f"- {label}: {summary.get(key, 0.0)}")
    summary_md.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Stage2S offline training runs")
    parser.add_argument("--run-dir", type=str, required=True)
    return parser


def main(argv=None) -> Dict:
    args = build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    summary = summarize_run(run_dir)
    write_summary_files(run_dir, summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

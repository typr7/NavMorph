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
    ("changed_rate", "Changed Rate"),
    ("progress_distance_corr", "Progress-vs-Distance Corr"),
]


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


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


def _kill_checks(summary: Dict) -> Dict[str, Dict[str, object]]:
    changed_rate = summary.get("changed_rate")
    exec_ece = summary.get("exec_ece")
    exec_auroc = summary.get("exec_auroc")
    progress_distance_corr = summary.get("progress_distance_corr")
    wider_search_only_gain = summary.get("wider_search_only_gain")

    checks = {
        "changed_rate_negligible": {
            "status": None if changed_rate is None else bool(changed_rate < 0.01),
            "value": changed_rate,
            "reason": "Flag if online intervention is still in the old near-zero regime.",
        },
        "exec_calibration_poor": {
            "status": None if exec_ece is None and exec_auroc is None else bool((exec_ece or 0.0) > 0.10 or (exec_auroc or 0.0) < 0.65),
            "value": {"exec_ece": exec_ece, "exec_auroc": exec_auroc},
            "reason": "Flag if executability confidence is poorly calibrated or weakly discriminative.",
        },
        "progress_distance_shortcut": {
            "status": None if progress_distance_corr is None else bool(abs(progress_distance_corr) > 0.60),
            "value": progress_distance_corr,
            "reason": "Flag if progress prediction still tracks raw distance too strongly.",
        },
        "gains_only_from_search": {
            "status": None if wider_search_only_gain is None else bool(wider_search_only_gain),
            "value": wider_search_only_gain,
            "reason": "Flag if gains only appear after widening search instead of improving branch scoring.",
        },
    }
    return checks


def summarize_run(run_dir: Path) -> Dict:
    history = _load_history(run_dir)
    run_summary = _load_json(run_dir / "run_summary.json")
    debug_metrics = _load_json(run_dir / "debug_step_metrics.json")

    if history:
        best = _best_epoch_record(history)
        summary = {
            "best_epoch": best.get("epoch"),
            "best_total_loss": best.get("val_total_loss", best.get("train_total_loss")),
        }
        for key, _ in SUMMARY_KEYS:
            value = best.get(f"val_{key}", best.get(f"train_{key}"))
            if value is None:
                value = run_summary.get(key)
            summary[key] = value
    else:
        summary = {
            "best_epoch": 0,
            "best_total_loss": debug_metrics.get("total_loss", 0.0),
            "exec_auroc": run_summary.get("exec_auroc", 0.5),
            "exec_ap": run_summary.get("exec_ap", 0.0),
            "blocking_auroc": run_summary.get("blocking_auroc", 0.5),
            "blocking_ap": run_summary.get("blocking_ap", 0.0),
            "progress_mae": run_summary.get("progress_mae", 0.0),
            "progress_corr": run_summary.get("progress_corr", 0.0),
            "ranking_top1_acc": run_summary.get("ranking_top1_acc", 0.0),
            "exec_ece": run_summary.get("exec_ece", 0.0),
            "blocking_ece": run_summary.get("blocking_ece", 0.0),
            "changed_rate": run_summary.get("changed_rate"),
            "progress_distance_corr": run_summary.get("progress_distance_corr"),
        }

    for key in ("changed_rate", "progress_distance_corr", "wider_search_only_gain"):
        if key not in summary and key in run_summary:
            summary[key] = run_summary[key]

    summary["kill_checks"] = _kill_checks(summary)
    return summary


def write_summary_files(run_dir: Path, summary: Dict) -> None:
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Stage2S Offline Summary",
        "",
        f"- Best epoch: {summary['best_epoch']}",
        f"- Best total loss: {summary['best_total_loss']}",
    ]
    for key, label in SUMMARY_KEYS:
        lines.append(f"- {label}: {summary.get(key, None)}")

    lines.extend(["", "## Kill Checks"])
    for key, payload in summary.get("kill_checks", {}).items():
        lines.append(f"- {key}: status={payload.get('status')} value={payload.get('value')} — {payload.get('reason')}")
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

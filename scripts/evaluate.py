"""
scripts/evaluate.py

Lightweight evaluation harness that replays a dataset (or synthetic samples)
through the agent pipeline and reports precision/recall on the detected issues.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

from agents.orchestrator import AgentOrchestrator
from services.session_service import InMemorySessionService
from utils.observability import setup_logging, log_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Table Tennis Coach agents.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to JSON manifest with entries: "
        "[{'video': '/path/to.mp4', 'expected_issues': [...], 'user_profile': {...}}]",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run evaluation against synthetic detections (no video required).",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list of samples.")
    return data


def collect_issues(evaluations: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for evaluation in evaluations:
        issues.extend(evaluation.get("issues", []))
    return issues


def compute_metrics(predicted: List[str], expected: List[str]) -> Tuple[float, float, float]:
    pred_set = set(predicted)
    exp_set = set(expected)
    true_positive = len(pred_set & exp_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(exp_set) if exp_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run_manifest_evaluation(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    session_service = InMemorySessionService()
    orchestrator = AgentOrchestrator(session_service)
    per_sample = []
    for sample in samples:
        session = session_service.ensure_session(
            None,
            user_profile=sample.get("user_profile") or {"level": "Intermediate"},
        )
        start = time.time()
        result = orchestrator.run(sample["video"], session, resume=False)
        latency = time.time() - start
        predicted_issues = collect_issues(result["evaluations"])
        precision, recall, f1 = compute_metrics(predicted_issues, sample.get("expected_issues", []))
        per_sample.append(
            {
                "session_id": result["session_id"],
                "video": sample["video"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency_sec": latency,
                "predicted_issues": predicted_issues,
                "expected_issues": sample.get("expected_issues", []),
            }
        )
    macro_precision = sum(s["precision"] for s in per_sample) / len(per_sample) if per_sample else 0.0
    macro_recall = sum(s["recall"] for s in per_sample) / len(per_sample) if per_sample else 0.0
    macro_f1 = sum(s["f1"] for s in per_sample) / len(per_sample) if per_sample else 0.0
    return {
        "samples": per_sample,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def run_mock_evaluation() -> Dict[str, Any]:
    """Fallback evaluation that uses synthetic detections to validate scoring logic."""
    synthetic_evaluations = [
        {
            "frames": list(range(10)),
            "score": 70,
            "issues": ["Racket angle undetected in this sequence", "Elbow appears high for some frames (may reduce control)"],
        },
        {
            "frames": list(range(10, 20)),
            "score": 95,
            "issues": [],
        },
    ]
    predicted = collect_issues(synthetic_evaluations)
    expected = ["Racket angle undetected in this sequence"]
    precision, recall, f1 = compute_metrics(predicted, expected)
    return {
        "samples": [
            {
                "session_id": "synthetic",
                "video": "synthetic",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_issues": predicted,
                "expected_issues": expected,
            }
        ],
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def save_report(report: Dict[str, Any]) -> Path:
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"evaluation_{int(time.time())}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main():
    args = parse_args()
    setup_logging()
    if args.mock:
        report = run_mock_evaluation()
    elif args.manifest:
        samples = load_manifest(args.manifest)
        report = run_manifest_evaluation(samples)
    else:
        raise SystemExit("Provide --manifest /path/to.json or --mock.")
    log_event(
        "evaluation_complete",
        macro_precision=report["macro_precision"],
        macro_recall=report["macro_recall"],
        macro_f1=report["macro_f1"],
    )
    report_path = save_report(report)
    print(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from debate_sim.config import AppConfig
from debate_sim.factory import build_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debate Simulation & Evaluation Engine")
    parser.add_argument("--topic", required=True, help="Debate topic")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of simulation runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    engine = build_engine(config)
    report = engine.run_simulation(topic=args.topic, num_runs=args.num_runs)

    print(json.dumps(
        {
            "topic": report["topic"],
            "num_runs": report["num_runs"],
            "provider": config.provider,
            "model": config.model,
            "max_workers": config.max_workers,
            "wins": report["wins"],
            "completed_win_rate": report.get("completed_win_rate", report.get("win_rate")),
            "completed_runs": report.get("completed_runs", report["num_runs"]),
            "failed_runs_count": report.get("failed_runs_count", 0),
            "top3_pro_arguments": report["top3_pro_arguments"],
            "top3_con_arguments": report["top3_con_arguments"],
            "top10_pro_arguments": report["top10_pro_arguments"],
            "top10_con_arguments": report["top10_con_arguments"],
            "best_argument_combinations": report["best_argument_combinations"],
            "selection_mode_summary": report.get("selection_mode_summary", {}),
            "top_key_clashes": report["top_key_clashes"],
            "output": str((config.results_dir / "simulation_report.json")),
            "markdown_output": str((config.results_dir / "simulation_report.md")),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

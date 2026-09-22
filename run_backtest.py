from __future__ import annotations

import argparse
from pathlib import Path

from phase1_backtest import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 backtesting pipeline")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download a fresh Yahoo Finance snapshot instead of using the local audited cache",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    result = run_pipeline(project_root, refresh_data=args.refresh_data)
    metrics = result["metrics"]
    print(f"Completed {len(metrics)} strategy/ticker evaluations using {result['source_mode']} data.")
    print(f"Results: {project_root / 'results' / 'performance_metrics.csv'}")
    print(f"Report:  {project_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()


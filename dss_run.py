"""
DSS Runner — CLI entry point.
Usage:
  python dss_run.py --input match.json --output results.json
  python dss_run.py --input match.json --output results.json --strategies custom_strategies.json
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError
from dss_schema import DSSInput, DSSOutput
from dss_engine import run_batch, load_strategies


def main():
    parser = argparse.ArgumentParser(description="Football DSS - Strategy Recommender")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--strategies", default=None, help="Path to strategy_templates.json (optional)")
    args = parser.parse_args()

    # --- Load input file ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    # --- Validate input ---
    try:
        validated = DSSInput(**raw_data)
    except ValidationError as e:
        print("[ERROR] Input validation failed:", file=sys.stderr)
        for err in e.errors():
            loc = " → ".join(str(x) for x in err["loc"])
            print(f"  {loc}: {err['msg']}", file=sys.stderr)
        sys.exit(1)

    # --- Load strategies ---
    strategies = None
    if args.strategies:
        strategies = load_strategies(args.strategies)

    # --- Run engine ---
    output_data = run_batch(validated.model_dump(), strategies)

    # --- Validate output (safety check) ---
    output_validated = DSSOutput(**output_data)

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_validated.model_dump(), f, indent=2)

    n = output_validated.meta.total_scenarios
    best = output_validated.results[0].best_strategy.strategy
    print(f"[OK] {n} scenarios processed. Output: {output_path}")
    print(f"     First scenario best strategy: {best}")


if __name__ == "__main__":
    main()

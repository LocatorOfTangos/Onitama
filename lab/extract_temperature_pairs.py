#!/usr/bin/env python3
"""
Extract iteration-temperature pairs from pmcts_temperature_history.json.

Reads the JSON history and writes a simple CSV file with two columns:
iteration, temperature
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def extract_temperature_pairs(input_path, output_path):
    """Read JSON history and write iteration,temperature CSV."""
    with open(input_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("iteration,temperature\n")
        for entry in history:
            iteration = entry["iteration"]
            temperature = entry["temperature"]
            f.write(f"{iteration},{temperature}\n")
    
    print(f"Extracted {len(history)} iteration-temperature pairs")
    print(f"Written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract iteration-temperature pairs from PMCTS tuning history"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "pmcts_temperature_history.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "temperature_pairs.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    extract_temperature_pairs(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())

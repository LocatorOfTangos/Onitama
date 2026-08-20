#!/usr/bin/env python3
"""Analyze NN training examples and export metric plots.

This script loads all generation tensors in a folder (default:
src/onitama/opponents/NN/logs/vn0_generations), aggregates all examples,
and computes:
  - value-target histogram + mean/stdev
  - histogram of active-side student counts
  - histogram of (active students - opponent students)
  - per-card total presence counts (bar chart)
  - per-active-hand-card-pair total counts (bar chart)
    - count of initial-position examples that are duplicated by card arrangement

Card and pair charts are bar charts because this workflow aggregates all
generation files into one total per category.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from onitama.Deck import Deck


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "src" / "onitama" / "opponents" / "NN" / "logs" / "vn0_generations"
DEFAULT_OUTPUT_DIR = ROOT / "results"


def card_names_by_index() -> list[str]:
    names = [""] * len(Deck.DECK)
    for card in Deck.DECK:
        names[card.idx] = card.name
    return names


def load_generation_tensors(input_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Path]]:
    files = sorted(input_dir.glob("gen_*.pt"))
    if not files:
        files = sorted(input_dir.glob("prepped_gen*.pt"))
    if not files:
        raise FileNotFoundError(f"No generation files found in: {input_dir}")

    boards_list: list[torch.Tensor] = []
    cards_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []

    for path in files:
        payload = torch.load(path, map_location="cpu")
        if not all(key in payload for key in ("boards", "cards", "values")):
            raise KeyError(f"Missing one of required keys boards/cards/values in {path}")

        boards = payload["boards"]
        cards = payload["cards"]
        values = payload["values"]

        if boards.ndim != 4 or boards.shape[1:] != (4, 5, 5):
            raise ValueError(f"Unexpected boards shape in {path}: {tuple(boards.shape)}")
        if cards.ndim != 3 or cards.shape[1:] != (3, 16):
            raise ValueError(f"Unexpected cards shape in {path}: {tuple(cards.shape)}")
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(f"Unexpected values shape in {path}: {tuple(values.shape)}")

        n = boards.shape[0]
        if cards.shape[0] != n or values.shape[0] != n:
            raise ValueError(
                f"Mismatched batch sizes in {path}: boards={boards.shape[0]}, cards={cards.shape[0]}, values={values.shape[0]}"
            )

        boards_list.append(boards)
        cards_list.append(cards)
        values_list.append(values)

    return torch.cat(boards_list, dim=0), torch.cat(cards_list, dim=0), torch.cat(values_list, dim=0), files


def save_histogram(data: np.ndarray, bins: int, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_bar_chart(labels: list[str], counts: list[int], title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(labels)), 6.5))
    x = np.arange(len(labels))
    ax.bar(x, counts)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Position Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_counts_csv(rows: list[tuple[str, int]], out_path: Path, header_name: str) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header_name, "count"])
        for name, count in rows:
            writer.writerow([name, count])


def build_initial_position_templates(device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    templates: list[torch.Tensor] = []

    for friendly_home_y, enemy_home_y in ((4, 0), (0, 4)):
        template = torch.zeros((4, 5, 5), dtype=dtype, device=device)

        # Active side (friendly) home row.
        template[0, friendly_home_y, 2] = 1.0
        template[1, friendly_home_y, 0] = 1.0
        template[1, friendly_home_y, 1] = 1.0
        template[1, friendly_home_y, 3] = 1.0
        template[1, friendly_home_y, 4] = 1.0

        # Opponent home row.
        template[2, enemy_home_y, 2] = 1.0
        template[3, enemy_home_y, 0] = 1.0
        template[3, enemy_home_y, 1] = 1.0
        template[3, enemy_home_y, 3] = 1.0
        template[3, enemy_home_y, 4] = 1.0

        templates.append(template)

    return templates


def count_initial_position_duplicate_examples(boards: torch.Tensor, cards: torch.Tensor) -> dict[str, int]:
    initial_templates = build_initial_position_templates(device=boards.device, dtype=boards.dtype)
    initial_masks = [(boards == template.unsqueeze(0)).all(dim=(1, 2, 3)) for template in initial_templates]
    is_initial_position = torch.stack(initial_masks, dim=0).any(dim=0)

    initial_cards = cards[is_initial_position]
    arrangement_counter: Counter[tuple[int, ...]] = Counter()

    for card_tensor in initial_cards:
        key = tuple(torch.nonzero(card_tensor, as_tuple=False).flatten().tolist())
        arrangement_counter[key] += 1

    duplicated_example_count = sum(count for count in arrangement_counter.values() if count >= 2)
    duplicated_arrangement_count = sum(1 for count in arrangement_counter.values() if count >= 2)

    return {
        "initial_position_examples": int(initial_cards.shape[0]),
        "initial_position_duplicated_examples": int(duplicated_example_count),
        "initial_position_duplicated_arrangements": int(duplicated_arrangement_count),
    }


def analyze_examples(input_dir: Path, output_dir: Path, bins: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    boards, cards, values, files = load_generation_tensors(input_dir)
    card_names = card_names_by_index()

    values_np = values.squeeze(1).numpy()
    values_mean = float(values_np.mean())
    values_std = float(values_np.std())

    friendly_students = boards[:, 1, :, :].sum(dim=(1, 2)).numpy()
    enemy_students = boards[:, 3, :, :].sum(dim=(1, 2)).numpy()
    student_diff = friendly_students - enemy_students
    initial_duplicate_stats = count_initial_position_duplicate_examples(boards, cards)

    card_presence_any = (cards > 0).any(dim=1)
    card_counts = card_presence_any.sum(dim=0).to(torch.int64).tolist()
    card_rows = list(zip(card_names, card_counts))
    card_rows_sorted = sorted(card_rows, key=lambda item: item[1], reverse=True)

    pair_counter: Counter[tuple[int, int]] = Counter()
    active_hand_presence = cards[:, 0, :] > 0
    for row in active_hand_presence:
        indices = torch.nonzero(row, as_tuple=False).squeeze(1).tolist()
        if len(indices) < 2:
            continue
        for i, j in combinations(sorted(indices), 2):
            pair_counter[(i, j)] += 1

    pair_rows = [
        (f"{card_names[i]} + {card_names[j]}", count)
        for (i, j), count in pair_counter.items()
    ]
    pair_rows_sorted = sorted(pair_rows, key=lambda item: item[1], reverse=True)

    save_histogram(
        data=values_np,
        bins=bins,
        title="Distribution of Value Targets",
        xlabel="Target Value",
        out_path=output_dir / "nn_value_target_hist.png",
    )
    save_histogram(
        data=friendly_students,
        bins=np.arange(-0.5, 6.5, 1.0).size - 1,
        title="Distribution of Active-Player Student Count",
        xlabel="Active-Player Students Alive",
        out_path=output_dir / "nn_active_student_count_hist.png",
    )
    save_histogram(
        data=student_diff,
        bins=np.arange(-5.5, 6.5, 1.0).size - 1,
        title="Distribution of Student Count Difference",
        xlabel="Active Students - Opponent Students",
        out_path=output_dir / "nn_student_diff_hist.png",
    )

    if card_rows_sorted:
        labels, counts = zip(*card_rows_sorted)
        save_bar_chart(
            labels=list(labels),
            counts=list(counts),
            title="Total Positions Featuring Each Card (Any Slot)",
            xlabel="Card",
            out_path=output_dir / "nn_card_presence_counts.png",
        )
        save_counts_csv(card_rows_sorted, output_dir / "nn_card_presence_counts.csv", "card")

    if pair_rows_sorted:
        pair_labels, pair_counts = zip(*pair_rows_sorted)
        save_bar_chart(
            labels=list(pair_labels),
            counts=list(pair_counts),
            title="Total Positions Featuring Each Active-Hand Card Pair",
            xlabel="Active-Hand Card Pair",
            out_path=output_dir / "nn_active_hand_pair_counts.png",
        )
        save_counts_csv(pair_rows_sorted, output_dir / "nn_active_hand_pair_counts.csv", "active_hand_pair")

    summary = {
        "input_dir": str(input_dir),
        "generation_files": [p.name for p in files],
        "num_generation_files": len(files),
        "num_examples": int(values.shape[0]),
        "value_target": {
            "mean": values_mean,
            "stdev": values_std,
            "min": float(values_np.min()),
            "max": float(values_np.max()),
        },
        "active_student_count": {
            "mean": float(friendly_students.mean()),
            "stdev": float(friendly_students.std()),
            "min": int(friendly_students.min()),
            "max": int(friendly_students.max()),
        },
        "student_diff": {
            "mean": float(student_diff.mean()),
            "stdev": float(student_diff.std()),
            "min": int(student_diff.min()),
            "max": int(student_diff.max()),
        },
        "initial_position_duplicate_card_arrangements": initial_duplicate_stats,
        "outputs": {
            "value_hist": "nn_value_target_hist.png",
            "active_student_hist": "nn_active_student_count_hist.png",
            "student_diff_hist": "nn_student_diff_hist.png",
            "card_counts_chart": "nn_card_presence_counts.png",
            "card_counts_csv": "nn_card_presence_counts.csv",
            "active_pair_chart": "nn_active_hand_pair_counts.png",
            "active_pair_csv": "nn_active_hand_pair_counts.csv",
        },
    }

    with (output_dir / "nn_example_metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NN training examples from generation files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing gen_*.pt training example files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and summaries are written",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for value target histogram",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_examples(args.input_dir, args.output_dir, args.bins)
    print("NN example metrics complete")
    print(f"Input dir: {summary['input_dir']}")
    print(f"Generation files: {summary['num_generation_files']}")
    print(f"Examples: {summary['num_examples']}")
    print(f"Value mean/stdev: {summary['value_target']['mean']:.6f} / {summary['value_target']['stdev']:.6f}")
    print(
        "Initial-position examples with duplicated card arrangement: "
        f"{summary['initial_position_duplicate_card_arrangements']['initial_position_duplicated_examples']} "
        f"(of {summary['initial_position_duplicate_card_arrangements']['initial_position_examples']} initial-position examples)"
    )
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

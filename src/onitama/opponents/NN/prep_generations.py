#!/usr/bin/env python3
"""Prepare NN generation tensors for training.

This script:
1) Loads all gen_*.pt files from an input directory.
2) Deduplicates examples by position (board tensor + cards tensor), keeping
   the first target value encountered for each unique position.
3) Applies horizontal-flip augmentation using VN1_train.py logic.
4) Saves the prepared tensors into sharded .pt files.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from onitama.opponents.NN.VN1_train import build_card_flip_map, flip_cards_tensor


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "logs" / "vn_generations"
DEFAULT_OUTPUT_DIR = ROOT / "prepped_generations"
DEFAULT_MAX_SHARD_BYTES = 1_000_000_000


class ShardWriter:
    def __init__(self, output_dir: Path, max_shard_bytes: int) -> None:
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._boards: list[torch.Tensor] = []
        self._cards: list[torch.Tensor] = []
        self._values: list[float] = []
        self._estimated_bytes = 0
        self._shard_index = 1

        self.shard_paths: list[Path] = []
        self.examples_written = 0

    def _estimate_example_bytes(self, board: torch.Tensor, cards: torch.Tensor) -> int:
        board_bytes = board.numel() * board.element_size()
        cards_bytes = cards.numel() * cards.element_size()
        value_bytes = 4
        return board_bytes + cards_bytes + value_bytes

    def add(self, board: torch.Tensor, cards: torch.Tensor, value: float) -> None:
        board_detached = board.contiguous().clone()
        cards_detached = cards.contiguous().clone()

        self._boards.append(board_detached)
        self._cards.append(cards_detached)
        self._values.append(float(value))
        self._estimated_bytes += self._estimate_example_bytes(board_detached, cards_detached)

        if self._estimated_bytes >= self.max_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._boards:
            return

        boards = torch.stack(self._boards, dim=0)
        cards = torch.stack(self._cards, dim=0)
        values = torch.tensor(self._values, dtype=torch.float32).unsqueeze(1)

        shard_path = self.output_dir / f"gen_{self._shard_index:02d}.pt"
        torch.save(
            {
                "boards": boards,
                "cards": cards,
                "values": values,
            },
            shard_path,
        )

        self.examples_written += int(boards.shape[0])
        self.shard_paths.append(shard_path)
        self._shard_index += 1

        self._boards.clear()
        self._cards.clear()
        self._values.clear()
        self._estimated_bytes = 0


def position_digest(board: torch.Tensor, cards: torch.Tensor) -> bytes:
    board_np = board.contiguous().numpy()
    cards_np = cards.contiguous().numpy()

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(board_np.tobytes())
    hasher.update(cards_np.tobytes())
    return hasher.digest()


def list_generation_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("gen_*.pt"))
    if not files:
        raise FileNotFoundError(f"No generation files found in: {input_dir}")
    return files


def validate_payload_shapes(path: Path, boards: torch.Tensor, cards: torch.Tensor, values: torch.Tensor) -> None:
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


def add_if_new(
    writer: ShardWriter,
    seen_augmented: set[bytes],
    board: torch.Tensor,
    cards: torch.Tensor,
    value: float,
) -> bool:
    key = position_digest(board, cards)
    if key in seen_augmented:
        return False

    seen_augmented.add(key)
    writer.add(board, cards, value)
    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load generation tensors, remove duplicate positions, "
            "apply horizontal flip augmentation, and save sharded .pt files"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing gen_*.pt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for sharded files (default: src/onitama/opponents/NN/prepped_generations)",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
        help="Approximate max raw tensor bytes per output shard (default: 1000000000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be > 0")

    files = list_generation_files(args.input_dir)

    flip_map = build_card_flip_map()
    writer = ShardWriter(args.output_dir, args.max_shard_bytes)

    seen_raw_value_by_key: dict[bytes, float] = {}
    seen_augmented: set[bytes] = set()

    raw_count = 0
    unique_count = 0
    duplicates_removed = 0
    conflicting_target_duplicates = 0
    post_augmentation_duplicates_removed = 0

    for path in files:
        payload = torch.load(path, map_location="cpu")
        if not all(key in payload for key in ("boards", "cards", "values")):
            raise KeyError(f"Missing one of required keys boards/cards/values in {path}")

        boards = payload["boards"]
        cards = payload["cards"]
        values = payload["values"]
        validate_payload_shapes(path, boards, cards, values)

        for idx in range(boards.shape[0]):
            raw_count += 1

            board = boards[idx]
            cards_tensor = cards[idx]
            value = float(values[idx, 0].item())

            raw_key = position_digest(board, cards_tensor)
            if raw_key in seen_raw_value_by_key:
                duplicates_removed += 1
                if abs(value - seen_raw_value_by_key[raw_key]) > 1e-8:
                    conflicting_target_duplicates += 1
                continue

            seen_raw_value_by_key[raw_key] = value
            unique_count += 1

            if not add_if_new(writer, seen_augmented, board, cards_tensor, value):
                post_augmentation_duplicates_removed += 1

            flipped_board = board.flip(-1)
            flipped_cards = flip_cards_tensor(cards_tensor, flip_map)
            if not add_if_new(writer, seen_augmented, flipped_board, flipped_cards, value):
                post_augmentation_duplicates_removed += 1

    writer.flush()

    final_output_count = writer.examples_written

    print("Prepared generations complete")
    print(f"Input dir: {args.input_dir}")
    print(f"Generation files: {len(files)}")
    print(f"Raw examples: {raw_count}")
    print(f"Unique examples: {unique_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Conflicting target duplicates: {conflicting_target_duplicates}")
    print(f"Post-augmentation duplicates removed: {post_augmentation_duplicates_removed}")
    print(f"Output examples (unique + flipped, deduped): {final_output_count}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output shards: {len(writer.shard_paths)}")
    if writer.shard_paths:
        print(f"First shard: {writer.shard_paths[0]}")
        print(f"Last shard: {writer.shard_paths[-1]}")


if __name__ == "__main__":
    main()

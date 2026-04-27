"""Shared RNG helpers for humaneval / vlm_test style scripts."""
from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def shuffled_copy(items: Sequence[T], *, seed: int | None = None) -> list[T]:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def stratified_halves(items: Sequence[T], *, seed: int | None = None) -> tuple[list[T], list[T]]:
    """Split into two groups of equal size (``len`` must be even)."""
    s = shuffled_copy(items, seed=seed)
    mid = len(s) // 2
    return s[:mid], s[mid:]

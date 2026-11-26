"""
Deterministic, stratified tile-based Train/Validation splitting utilities.
Groups windows by geographic tile (preventing spatial/window overlap leakage)
and stratifies validation tiles across dominant Köppen climate zones.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


def get_tile_dominant_climate(npz_path: Union[str, Path]) -> str:
    """
    Returns the most frequent Köppen climate zone for a tile.
    Reads the sibling .json metadata file first (fast), falling back to the .npz archive.
    """
    npz_path = Path(npz_path)
    json_path = npz_path.with_suffix(".json")

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        records = meta.get("metadata_records", [])
        if records:
            counts = Counter(str(r.get("climate") or "unknown").strip() or "unknown" for r in records)
            return counts.most_common(1)[0][0]

    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        if "climate" in d and len(d["climate"]) > 0:
            counts = Counter(str(c or "unknown").strip() or "unknown" for c in d["climate"])
            return counts.most_common(1)[0][0]

    return "unknown"


def get_train_val_tile_split(
    data_dir: Union[str, Path],
    num_val_tiles: int = 14,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], Dict[str, Dict[str, int]]]:
    """
    Splits .npz tiles in `data_dir` into disjoint train and validation tile lists,
    stratified by each tile's dominant Köppen climate zone.

    Singleton climate groups (count == 1) always stay in train so training never
    loses representation of a climate zone.
    """
    data_dir = Path(data_dir)
    all_files = sorted(list(data_dir.glob("*.npz")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if num_val_tiles <= 0 or num_val_tiles >= len(all_files):
        raise ValueError(
            f"num_val_tiles must be in [1, {len(all_files) - 1}], got {num_val_tiles}."
        )

    climate_groups: Dict[str, List[Path]] = defaultdict(list)
    for fpath in all_files:
        climate_groups[get_tile_dominant_climate(fpath)].append(fpath)

    rng = np.random.RandomState(seed)
    sorted_climates = sorted(climate_groups.keys())
    shuffled_groups: Dict[str, List[Path]] = {}
    for clim in sorted_climates:
        files = list(climate_groups[clim])
        perm = rng.permutation(len(files))
        shuffled_groups[clim] = [files[i] for i in perm]

    # Eligible groups have at least 2 tiles so at least 1 always remains in train
    eligible_climates = [c for c in sorted_climates if len(shuffled_groups[c]) >= 2]
    max_possible_val = sum(len(shuffled_groups[c]) - 1 for c in eligible_climates)
    if num_val_tiles > max_possible_val:
        raise ValueError(
            f"Cannot allocate {num_val_tiles} validation tiles while preserving at least "
            f"1 train tile per climate (max possible: {max_possible_val})."
        )

    total_eligible = sum(len(shuffled_groups[c]) for c in eligible_climates)
    val_alloc: Dict[str, int] = {c: 0 for c in sorted_climates}
    remainders: List[Tuple[float, int, str]] = []

    for clim in eligible_climates:
        n = len(shuffled_groups[clim])
        exact = (num_val_tiles * n) / total_eligible
        base = min(n - 1, int(np.floor(exact)))
        val_alloc[clim] = base
        remainders.append((exact - base, n, clim))

    remaining = num_val_tiles - sum(val_alloc.values())
    # Largest-remainder apportionment respecting the (n - 1) cap
    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
    while remaining > 0:
        allocated_in_pass = False
        for _, n, clim in remainders:
            if remaining == 0:
                break
            if val_alloc[clim] < n - 1:
                val_alloc[clim] += 1
                remaining -= 1
                allocated_in_pass = True
        if not allocated_in_pass:
            break

    train_files: List[Path] = []
    val_files: List[Path] = []
    train_climate_counts: Dict[str, int] = {}
    val_climate_counts: Dict[str, int] = {}

    for clim in sorted_climates:
        files = shuffled_groups[clim]
        k = val_alloc[clim]
        v_files = files[:k]
        t_files = files[k:]
        val_files.extend(v_files)
        train_files.extend(t_files)
        if t_files:
            train_climate_counts[clim] = len(t_files)
        if v_files:
            val_climate_counts[clim] = len(v_files)

    train_files.sort(key=lambda p: p.name)
    val_files.sort(key=lambda p: p.name)

    summary = {
        "train": train_climate_counts,
        "val": val_climate_counts,
    }
    return train_files, val_files, summary

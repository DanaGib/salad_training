"""Ground truth helpers for the MSLS Challenge test set.

Reads subtask_index.csv and postprocessed.csv to build per-city GT,
then globally offsets DB indices for concatenated multi-city datasets.
"""
import csv
import warnings
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree

THRESHOLD_M = 25.0
# Folder names on disk — buenosaires has no underscore.
CITIES = ["miami", "athens", "bengaluru", "buenosaires", "kampala", "stockholm"]


def read_keys(city_split_dir: Path, subtask: str = "all") -> list:
    """Return keys from subtask_index.csv where the subtask column is True.

    Falls back to all keys with a warning if no rows match.

    Args:
        city_split_dir: Path to a city's query/ or database/ metadata folder.
        subtask: Column name in subtask_index.csv to filter on.
    """
    path = city_split_dir / "subtask_index.csv"
    keys = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get(subtask, "").strip() == "True":
                keys.append(row["key"])
    if not keys:
        warnings.warn(f"No '{subtask}' entries in {path}; using all keys.")
        with open(path, newline="") as f:
            keys = [row["key"] for row in csv.DictReader(f)]
    return keys


def read_utm(city_split_dir: Path, keys: list) -> tuple:
    """Return (easting, northing) arrays for the given keys from postprocessed.csv.

    Args:
        city_split_dir: Path to a city's query/ or database/ metadata folder.
        keys: Ordered list of image key strings.
    """
    coord_map = {}
    with open(city_split_dir / "postprocessed.csv", newline="") as f:
        for row in csv.DictReader(f):
            coord_map[row["key"]] = (float(row["easting"]), float(row["northing"]))
    e = np.array([coord_map[k][0] for k in keys])
    n = np.array([coord_map[k][1] for k in keys])
    return e, n


def city_gt(
    e_q: np.ndarray, n_q: np.ndarray,
    e_db: np.ndarray, n_db: np.ndarray,
    db_offset: int,
) -> list:
    """BallTree radius search within one city; returns global DB index arrays.

    Args:
        e_q, n_q: Query UTM coordinate arrays.
        e_db, n_db: Database UTM coordinate arrays.
        db_offset: Number of DB images preceding this city in the global list.
    """
    tree = BallTree(np.column_stack([e_db, n_db]), metric="euclidean")
    return [
        idx + db_offset
        for idx in tree.query_radius(np.column_stack([e_q, n_q]), r=THRESHOLD_M)
    ]

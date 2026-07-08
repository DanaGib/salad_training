"""Shared UTM ground-truth helpers for VPR evaluation datasets."""
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree


def parse_utm(filename: str) -> tuple:
    """Return (easting, northing) from an @-encoded VPR dataset filename.

    Both SVOX and SF-XL use the convention:
        @UTM_Easting@UTM_Northing@...
    Splits the basename on '@'; indices 1 and 2 carry easting and northing.

    Args:
        filename: Full path or bare filename with @-delimited fields.

    Returns:
        Tuple of (easting, northing) as floats.
    """
    parts = Path(filename).name.split("@")
    return float(parts[1]), float(parts[2])


def build_gt_utm(
    e_q: np.ndarray,
    n_q: np.ndarray,
    e_db: np.ndarray,
    n_db: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    """Return per-query arrays of matching DB indices within threshold_m metres.

    Uses a Euclidean BallTree on raw UTM coordinates. This is valid when all
    images share the same UTM zone (planar distance == metric distance).

    Args:
        e_q: Query easting coordinates, shape (Q,).
        n_q: Query northing coordinates, shape (Q,).
        e_db: Database easting coordinates, shape (DB,).
        n_db: Database northing coordinates, shape (DB,).
        threshold_m: Radius in metres for a retrieval to be considered correct.

    Returns:
        Object array of shape (Q,) where gt[i] is an int array of DB indices
        within threshold_m of query i.
    """
    tree = BallTree(np.column_stack([e_db, n_db]), metric="euclidean")
    indices = tree.query_radius(np.column_stack([e_q, n_q]), r=threshold_m)
    gt = np.empty(len(e_q), dtype=object)
    for i, idx in enumerate(indices):
        gt[i] = idx
    return gt

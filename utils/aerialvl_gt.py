"""Ground-truth helpers for the AerialVL aerial VPR dataset.

AerialVL filenames encode WGS84 lon/lat coordinates directly (no UTM/XY
projection), so ground truth is built with a haversine-metric BallTree
instead of the Euclidean trees used for the UTM/XY-based datasets.

Reference: He et al., "AerialVL: A Dataset, Baseline and Algorithm
Framework for Aerial-Based Visual Localization With Reference Map",
IEEE RA-L 2024. https://github.com/hmf21/AerialVL
"""
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000.0


def parse_query_lonlat(filename: str) -> tuple:
    """Return (lon, lat) from an AerialVL query filename.

    Format: @Longitude@Latitude@.png — splits the basename on '@';
    indices 1 and 2 are longitude and latitude respectively.

    Args:
        filename: Full path or bare filename with @-delimited fields.

    Returns:
        Tuple of (lon, lat) as floats, in degrees.
    """
    parts = Path(filename).name.split("@")
    return float(parts[1]), float(parts[2])


def parse_db_tile_center(filename: str) -> tuple:
    """Return the (lon, lat) center of an AerialVL map-tile filename.

    Format: @map@LeftBottomLon@LeftBottomLat@RightTopLon@RightTopLat@.png
    — the tile center is the midpoint of the two corner points.

    Args:
        filename: Full path or bare filename with @-delimited fields.

    Returns:
        Tuple of (lon, lat) as floats, in degrees.
    """
    parts = Path(filename).name.split("@")
    lon1, lat1, lon2, lat2 = (float(p) for p in parts[2:6])
    return (lon1 + lon2) / 2.0, (lat1 + lat2) / 2.0


def build_gt_haversine(
    q_lonlat: np.ndarray, db_lonlat: np.ndarray, threshold_m: float
) -> np.ndarray:
    """Return per-query arrays of matching DB indices within threshold_m metres.

    Uses a haversine-metric BallTree on radian (lat, lon) pairs — the
    correct great-circle distance for raw WGS84 coordinates, as opposed to
    a planar Euclidean tree which would be inaccurate over larger extents.

    Args:
        q_lonlat: Query (lon, lat) positions in degrees, shape (Q, 2).
        db_lonlat: Database (lon, lat) positions in degrees, shape (DB, 2).
        threshold_m: Radius in metres for a retrieval to be considered correct.

    Returns:
        Object array of shape (Q,) where gt[i] is an int array of DB indices
        within threshold_m of query i.
    """
    # BallTree's haversine metric expects (lat, lon) in radians.
    db_rad = np.radians(db_lonlat[:, [1, 0]])
    q_rad = np.radians(q_lonlat[:, [1, 0]])

    tree = BallTree(db_rad, metric="haversine")
    radius_rad = threshold_m / EARTH_RADIUS_M
    indices = tree.query_radius(q_rad, r=radius_rad)

    gt = np.empty(len(q_lonlat), dtype=object)
    for i, idx in enumerate(indices):
        gt[i] = idx
    return gt

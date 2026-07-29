"""Ground-truth helpers for the official Baidu Mall (IDL_dataset_cvpr17) dataset."""
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_cop(camera_file: Path) -> np.ndarray:
    """Return the (x, y, z) center-of-projection from a .camera pose file.

    File format (official IDL_dataset_cvpr17 layout):
        lines[0:3]  3x3 camera intrinsics matrix K
        lines[3]    "0 0 0"
        lines[4:7]  3x3 rotation matrix
        lines[-2]   center-of-projection "x y z"
        lines[-1]   "w h" image dimensions

    Args:
        camera_file: Path to a .camera pose file.

    Returns:
        (3,) array of (x, y, z) world coordinates.
    """
    with open(camera_file) as f:
        lines = f.readlines()
    return np.fromstring(lines[-2], dtype=float, sep=" ")


def assert_stems_match(image_files, gt_files) -> None:
    """Verify image/gt directory listings are index-aligned by filename stem.

    Args:
        image_files: Sorted list of image filenames.
        gt_files: Sorted list of .camera filenames, expected in the same order.

    Raises:
        ValueError: If any index has mismatched stems (i.e. the two directory
            listings are not aligned and positional pairing would be wrong).
    """
    for img_f, gt_f in zip(image_files, gt_files):
        if Path(img_f).stem != Path(gt_f).stem:
            raise ValueError(
                f"Image/GT filename mismatch: {img_f} vs {gt_f}. "
                "Directory listings are not index-aligned."
            )


def build_gt_xyz(q_xyz: np.ndarray, db_xyz: np.ndarray, threshold_m: float) -> np.ndarray:
    """Return per-query arrays of matching DB indices within threshold_m metres.

    Mirrors AnyLoc's default protocol: a plain 3D Euclidean NearestNeighbors
    radius search on (x, y, z) camera positions, with no angular filtering.

    Args:
        q_xyz: Query (x, y, z) positions, shape (Q, 3).
        db_xyz: Database (x, y, z) positions, shape (DB, 3).
        threshold_m: Radius in metres for a retrieval to be considered correct.

    Returns:
        Object array of shape (Q,) where gt[i] is an int array of DB indices
        within threshold_m of query i.
    """
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(db_xyz)
    indices = knn.radius_neighbors(q_xyz, radius=threshold_m, return_distance=False)
    gt = np.empty(len(q_xyz), dtype=object)
    for i, idx in enumerate(indices):
        gt[i] = idx
    return gt

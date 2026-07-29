"""Baidu Mall (official IDL_dataset_cvpr17) dataset loader for VPR evaluation.

Uses the official CVPR'17 IDL release (689 database / 2292 query images)
with full 3D ground-truth camera positions parsed from .camera pose files,
mirroring AnyLoc's default protocol: NearestNeighbors radius search on
(x, y, z) center-of-projection coordinates, threshold 10 m, no angular
filtering. See utils/baidu_original_gt.py for the GT-building helpers.

Default threshold 10 m matches AnyLoc / SegVLAD / VLAD-BuFF for this split.
Override via BAIDU_ORIGINAL_THRESHOLD_M env var or the threshold_m constructor
arg.  Use 25 m to compare against VPR-methods-evaluation numbers on this split.

Dataset path: /home/eng/giborda/delavpr/salad/datasets/baidu_original/raw/
Layout:
    training_images_undistort/  -- 689 database images
    training_gt/                -- matching .camera pose files
    query_images_undistort/     -- 2292 query images
    query_gt/                   -- matching .camera pose files
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.baidu_original_gt import assert_stems_match, build_gt_xyz, parse_cop

BAIDU_ORIGINAL_ROOT = os.environ.get(
    "BAIDU_ORIGINAL_PATH",
    "/home/eng/giborda/delavpr/salad/datasets/baidu_original/raw/",
)
# 10 m matches AnyLoc/SegVLAD/VLAD-BuFF for the official 689/2292 split.
# Set BAIDU_ORIGINAL_THRESHOLD_M to override without editing source.
_DEFAULT_THRESHOLD_M = float(os.environ.get("BAIDU_ORIGINAL_THRESHOLD_M", 10.0))


class BaiduOriginalDataset(Dataset):
    """Official Baidu Mall (IDL_dataset_cvpr17) test set for VPR evaluation.

    Ground truth mirrors AnyLoc's default protocol: 3D Euclidean
    NearestNeighbors radius search on (x, y, z) camera positions
    parsed from .camera pose files, with no angular filtering.

    Args:
        input_transform: Optional torchvision transform pipeline.
        threshold_m: Positive-match radius in metres.  Defaults to
            BAIDU_ORIGINAL_THRESHOLD_M env var, then 10.0 (AnyLoc/SegVLAD
            standard for this split).  Use 25.0 to match VPR-methods-evaluation.
    """

    def __init__(self, input_transform=None, threshold_m: float = None):
        root = Path(BAIDU_ORIGINAL_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"Baidu original dataset not found at {BAIDU_ORIGINAL_ROOT}. "
                "Set BAIDU_ORIGINAL_PATH env var to the dataset root."
            )
        db_img_dir = root / "training_images_undistort"
        db_gt_dir = root / "training_gt"
        q_img_dir = root / "query_images_undistort"
        q_gt_dir = root / "query_gt"
        for d in (db_img_dir, db_gt_dir, q_img_dir, q_gt_dir):
            if not d.exists():
                raise FileNotFoundError(f"Expected directory not found: {d}")

        self.input_transform = input_transform
        threshold = threshold_m if threshold_m is not None else _DEFAULT_THRESHOLD_M

        db_files = sorted(os.listdir(db_img_dir))
        db_gt_files = sorted(os.listdir(db_gt_dir))
        q_files = sorted(os.listdir(q_img_dir))
        q_gt_files = sorted(os.listdir(q_gt_dir))

        assert_stems_match(db_files, db_gt_files)
        assert_stems_match(q_files, q_gt_files)

        self.dbImages = [str(db_img_dir / f) for f in db_files]
        self.qImages = [str(q_img_dir / f) for f in q_files]
        self.images = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        print(
            f"[BaiduOriginalDataset] split=IDL_cvpr17  "
            f"db={self.num_references}  q={self.num_queries}  "
            f"threshold={threshold}m  coord=XYZ-from-.camera"
        )

        db_xyz = np.array([parse_cop(db_gt_dir / f) for f in db_gt_files])
        q_xyz = np.array([parse_cop(q_gt_dir / f) for f in q_gt_files])
        self.ground_truth = build_gt_xyz(q_xyz, db_xyz, threshold)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

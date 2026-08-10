"""AerialVL aerial-based VPR dataset loader for evaluation.

Dataset path: /home/shared/datasets/AerialVL/images/VPR/
Layout:
    map_database/level_1|level_2|level_3/  -- satellite map tiles, named
        @map@LeftBottomLon@LeftBottomLat@RightTopLon@RightTopLat@.png
    query_images/query_images_1..4/        -- drone-captured frames, named
        @Longitude@Latitude@.png

Reference: He et al., "AerialVL: A Dataset, Baseline and Algorithm
Framework for Aerial-Based Visual Localization With Reference Map",
IEEE RA-L 2024. https://github.com/hmf21/AerialVL

Evaluation protocol (mirrors the AerialVL benchmark used by the LASED
paper, arXiv:2507.15089, Table II "AerialVL" column): the reference
database is the highest-altitude tile set (level_3) and the query set is
the union of all four query_images folders, with a positive match defined
as being within 50 m great-circle distance.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.aerialvl_gt import build_gt_haversine, parse_db_tile_center, parse_query_lonlat

AERIALVL_ROOT = os.environ.get(
    "AERIALVL_PATH",
    "/home/shared/datasets/AerialVL/images/VPR/",
)
AERIALVL_LEVEL = os.environ.get("AERIALVL_LEVEL", "level_3")
THRESHOLD_M = 50.0
QUERY_SUBDIR_PREFIX = "query_images_"


class AerialVLDataset(Dataset):
    """AerialVL dataset for VPR evaluation.

    Ground truth is built via a haversine-metric radius search (50 m) on
    WGS84 lon/lat coordinates parsed from filenames.

    Args:
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, input_transform=None):
        root = Path(AERIALVL_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"AerialVL dataset not found at {AERIALVL_ROOT}. "
                "Set AERIALVL_PATH env var to the dataset root."
            )
        db_dir = root / "map_database" / AERIALVL_LEVEL
        query_root = root / "query_images"
        if not db_dir.exists():
            raise FileNotFoundError(f"Expected directory not found: {db_dir}")
        if not query_root.exists():
            raise FileNotFoundError(f"Expected directory not found: {query_root}")

        query_dirs = sorted(
            d for d in query_root.iterdir()
            if d.is_dir() and d.name.startswith(QUERY_SUBDIR_PREFIX)
        )
        if not query_dirs:
            raise FileNotFoundError(
                f"No '{QUERY_SUBDIR_PREFIX}*' subfolders found under {query_root}"
            )

        self.input_transform = input_transform

        db_files = sorted(os.listdir(db_dir))
        q_files = []
        for qd in query_dirs:
            q_files.extend(str(qd / f) for f in sorted(os.listdir(qd)))

        self.dbImages = [str(db_dir / f) for f in db_files]
        self.qImages = q_files
        self.images = self.dbImages + self.qImages

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        db_lonlat = np.array([parse_db_tile_center(f) for f in db_files])
        q_lonlat = np.array([parse_query_lonlat(f) for f in self.qImages])

        self.ground_truth = build_gt_haversine(q_lonlat, db_lonlat, THRESHOLD_M)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

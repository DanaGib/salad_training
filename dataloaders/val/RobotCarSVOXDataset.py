"""RobotCar domain-query dataset loader using the SVOX gallery."""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils.utm_gt import build_gt_utm, parse_utm

ImageFile.LOAD_TRUNCATED_IMAGES = True

SVOX_ROOT = os.environ.get(
    "SVOX_PATH",
    "/home/eng/giborda/delavpr/datasets/SVOX/svox/images/",
)
THRESHOLD_M = 10.0
QUERY_SUBSETS = {"sun", "snow", "rain", "night", "overcast"}


class RobotCarSVOXDataset(Dataset):
    """RobotCar domain queries evaluated against the SVOX test gallery.

    Gallery:  test/gallery            (17,166 SVOX images, year 2012)
    Queries:  test/queries_<subset>   (RobotCar domain images)

    Available subsets and their sizes:
        sun       854 images  (2014)
        snow      870 images  (2015)
        rain      937 images  (2015)
        night     823 images  (2014)
        overcast  872 images  (2015)

    All 5 subsets share the same gallery, so when evaluated together in a
    single eval.py run the gallery descriptors are extracted only once (the
    extraction.py db_cache reuses them by num_references).

    Ground truth uses a 10 m BallTree threshold on UTM coordinates encoded in
    filenames (the threshold from the original WACV 2021 paper).

    Args:
        query_subset: One of "sun", "snow", "rain", "night", "overcast".
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, query_subset: str, input_transform=None):
        if query_subset not in QUERY_SUBSETS:
            raise ValueError(
                f"query_subset must be one of {QUERY_SUBSETS}, got '{query_subset}'"
            )
        root = Path(SVOX_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"SVOX dataset not found at {SVOX_ROOT}. "
                "Set SVOX_PATH env var to the dataset root."
            )
        db_dir = root / "test" / "gallery"
        q_dir = root / "test" / f"queries_{query_subset}"
        for d in (db_dir, q_dir):
            if not d.exists():
                raise FileNotFoundError(f"Expected directory not found: {d}")

        self.input_transform = input_transform

        db_files = sorted(os.listdir(db_dir))
        q_files = sorted(os.listdir(q_dir))

        self.dbImages = [str(db_dir / f) for f in db_files]
        self.qImages = [str(q_dir / f) for f in q_files]
        self.images = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        e_db = np.array([parse_utm(f)[0] for f in db_files])
        n_db = np.array([parse_utm(f)[1] for f in db_files])
        e_q = np.array([parse_utm(f)[0] for f in q_files])
        n_q = np.array([parse_utm(f)[1] for f in q_files])
        self.ground_truth = build_gt_utm(e_q, n_q, e_db, n_db, THRESHOLD_M)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

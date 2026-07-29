"""Baidu indoor localization dataset loader for VPR evaluation.

Dataset path: /home/shared/datasets/baidu/images/test/
Layout:
    database/   — 5,207 images with @x@y@-encoded local coordinates
    queries/    — 3,208 images with @x@y@-encoded local coordinates

Ground truth: Euclidean NearestNeighbors radius search on XY coordinates
parsed from filenames (local metric frame).  Default threshold 25 m matches
the VPR-methods-evaluation / Gabriele Berton benchmark for this split.
Override via BAIDU_THRESHOLD_M env var or the threshold_m constructor arg.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

BAIDU_ROOT = os.environ.get(
    "BAIDU_PATH",
    "/home/shared/datasets/baidu/images/test/",
)
# 25 m matches the VPR-methods-evaluation standard for the Berton split.
# Set BAIDU_THRESHOLD_M to override without editing source.
_DEFAULT_THRESHOLD_M = float(os.environ.get("BAIDU_THRESHOLD_M", 25.0))


def _parse_xy(filename: str) -> tuple:
    """Return (x, y) floats from @x@y@-encoded filename.

    Format: @x@y@...  — splits basename on '@'; indices 1 and 2 are x, y.

    Args:
        filename: Full path or bare filename with @-delimited fields.

    Returns:
        Tuple of (x, y) as floats.
    """
    parts = Path(filename).name.split("@")
    return float(parts[1]), float(parts[2])


class BaiduDataset(Dataset):
    """Baidu indoor test dataset for VPR evaluation (Berton/VPR-methods-evaluation split).

    DB and query images have XY coordinates encoded in their filenames.
    Ground truth is built via NearestNeighbors radius search.

    Args:
        input_transform: Optional torchvision transform pipeline.
        threshold_m: Positive-match radius in metres.  Defaults to
            BAIDU_THRESHOLD_M env var, then 25.0 (VPR-methods-evaluation
            standard for this split).  Use 10.0 to match AnyLoc/VLAD-BuFF.
    """

    def __init__(self, input_transform=None, threshold_m: float = None):
        root = Path(BAIDU_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"Baidu dataset not found at {BAIDU_ROOT}. "
                "Set BAIDU_PATH env var to the dataset root."
            )
        db_dir = root / "database"
        q_dir = root / "queries"
        for d in (db_dir, q_dir):
            if not d.exists():
                raise FileNotFoundError(f"Expected directory not found: {d}")

        self.input_transform = input_transform
        threshold = threshold_m if threshold_m is not None else _DEFAULT_THRESHOLD_M

        db_files = sorted(os.listdir(db_dir))
        q_files = sorted(os.listdir(q_dir))

        self.dbImages = [str(db_dir / f) for f in db_files]
        self.qImages = [str(q_dir / f) for f in q_files]
        self.images = self.dbImages + self.qImages

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        print(
            f"[BaiduDataset] split=Berton  "
            f"db={self.num_references}  q={self.num_queries}  "
            f"threshold={threshold}m  coord=XY-from-filename"
        )

        db_xy = np.array([_parse_xy(f) for f in db_files])
        q_xy = np.array([_parse_xy(f) for f in q_files])

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(db_xy)
        indices = knn.radius_neighbors(
            q_xy, radius=threshold, return_distance=False
        )
        self.ground_truth = np.empty(len(q_xy), dtype=object)
        for i, idx in enumerate(indices):
            self.ground_truth[i] = idx

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

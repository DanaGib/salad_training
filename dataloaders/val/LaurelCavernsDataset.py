"""Laurel Caverns dataset loader for VPR evaluation.

Dataset path: /home/shared/datasets/laurel_caverns/laurel_caverns/
Layout:
    db_images/           — 141 PNG images named by frame index (0.png … 140.png)
    q_images/            — 112 PNG images named by frame index (229.png … 340.png)
    pose_topic_list.npy  — shape (N, 7): columns 0-1 are XY position in metres

Ground truth: NearestNeighbors radius search on XY positions.
DB poses: rows [:141]; Q poses: rows [229:341].

The Q images start at frame 229 (not 141) because frames 141-228 are
transition/unused frames from the traversal sequence. The filenames encode
the frame index directly (229.png … 340.png), so sorting by integer stem
aligns files with the pose slice without any offset arithmetic.

WARNING: files must be sorted by integer stem. Plain sorted() is lexicographic
and would corrupt the pose-to-image mapping for multi-digit frame numbers.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

ImageFile.LOAD_TRUNCATED_IMAGES = True

LAUREL_ROOT = os.environ.get(
    "LAUREL_PATH",
    "/home/shared/datasets/laurel_caverns/laurel_caverns/",
)
THRESHOLD_M = 8.0
DB_POSE_SLICE = slice(0, 141)
Q_POSE_SLICE = slice(229, 341)


class LaurelCavernsDataset(Dataset):
    """Laurel Caverns dataset for VPR evaluation.

    Ground truth is built from XY coordinates in pose_topic_list.npy using
    a 8 m NearestNeighbors radius (matches original dataset reference).
    Q pose rows are non-contiguous with DB (gap at 141-228 are unused frames).

    Args:
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, input_transform=None):
        root = Path(LAUREL_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"Laurel Caverns dataset not found at {LAUREL_ROOT}. "
                "Set LAUREL_PATH env var to the dataset root."
            )
        db_dir = root / "db_images"
        q_dir = root / "q_images"
        pose_file = root / "pose_topic_list.npy"
        for p in (db_dir, q_dir, pose_file):
            if not p.exists():
                raise FileNotFoundError(f"Expected path not found: {p}")

        self.input_transform = input_transform

        # Sort by integer stem — critical for correct pose-to-image alignment.
        db_files = sorted(os.listdir(db_dir), key=lambda x: int(Path(x).stem))
        q_files = sorted(os.listdir(q_dir), key=lambda x: int(Path(x).stem))

        self.dbImages = [str(db_dir / f) for f in db_files]
        self.qImages = [str(q_dir / f) for f in q_files]
        self.images = self.dbImages + self.qImages

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        poses = np.load(pose_file)
        db_xy = poses[DB_POSE_SLICE, :2]
        q_xy = poses[Q_POSE_SLICE, :2]

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(db_xy)
        indices = knn.radius_neighbors(
            q_xy, radius=THRESHOLD_M, return_distance=False
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

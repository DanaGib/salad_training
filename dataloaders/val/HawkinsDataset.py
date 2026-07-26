"""Hawkins Long Corridor dataset loader for VPR evaluation.

Dataset path: /home/shared/datasets/hawkins/hawkins_long_corridor/
Layout:
    db_images/           — 127 PNG images named by frame index (0.png … 126.png)
    q_images/            — 118 PNG images named by frame index (127.png … 244.png)
    pose_topic_list.npy  — shape (N, 7): columns 0-1 are XY position in metres,
                           columns 2-6 are Z + quaternion (unused here)

Ground truth: NearestNeighbors radius search on XY positions from
pose_topic_list.npy. DB poses are rows [:127], Q poses are rows [127:245].
Files are sorted by integer stem to match the pose index order — critical
because plain lexicographic sort would corrupt the mapping.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

ImageFile.LOAD_TRUNCATED_IMAGES = True

HAWKINS_ROOT = os.environ.get(
    "HAWKINS_PATH",
    "/home/shared/datasets/hawkins/hawkins_long_corridor/",
)
THRESHOLD_M = 8.0
DB_POSE_SLICE = slice(0, 127)
Q_POSE_SLICE = slice(127, 245)


class HawkinsDataset(Dataset):
    """Hawkins Long Corridor dataset for VPR evaluation.

    Ground truth is built from XY coordinates in pose_topic_list.npy using
    a 8 m NearestNeighbors radius (matches original dataset reference).

    Args:
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, input_transform=None):
        root = Path(HAWKINS_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"Hawkins dataset not found at {HAWKINS_ROOT}. "
                "Set HAWKINS_PATH env var to the dataset root."
            )
        db_dir = root / "db_images"
        q_dir = root / "q_images"
        pose_file = root / "pose_topic_list.npy"
        for p in (db_dir, q_dir, pose_file):
            if not p.exists():
                raise FileNotFoundError(f"Expected path not found: {p}")

        self.input_transform = input_transform

        # Sort by integer stem — plain sorted() gives wrong lexicographic order
        # (e.g. 10.png before 2.png), which corrupts the pose-to-image mapping.
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

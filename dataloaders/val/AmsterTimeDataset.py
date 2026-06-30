import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

AMSTERTIME_ROOT = os.environ.get("AMSTERTIME_PATH", "../data/amstertime/")


class AmsterTimeDataset(Dataset):
    """AmsterTime evaluation dataset loader.

    Database: modern street-view images.
    Queries: historical photos of the same locations.
    Both directories contain identically-named files, so GT is 1-to-1:
    query[i] matches exactly database[i].

    Args:
        split: Currently only ``test`` is supported.
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, split: str = "test", input_transform=None):
        root = Path(AMSTERTIME_ROOT) / split
        if not root.exists():
            raise FileNotFoundError(f"AmsterTime split not found: {root}")

        self.input_transform = input_transform
        db_dir = root / "database"
        q_dir = root / "queries"

        db_files = sorted(os.listdir(db_dir))
        q_files = sorted(os.listdir(q_dir))

        if db_files != q_files:
            raise ValueError(
                "AmsterTime DB and query filenames do not match; cannot build 1:1 GT."
            )

        self.dbImages = [str(db_dir / f) for f in db_files]
        self.qImages = [str(q_dir / f) for f in q_files]
        self.images = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        # Ground truth: query i has exactly one correct match at database index i.
        self.ground_truth = np.array(
            [np.array([i]) for i in range(self.num_queries)], dtype=object
        )

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

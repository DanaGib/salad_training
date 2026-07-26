"""MeshVPR synthetic Melbourne dataset loader for VPR evaluation.

Dataset path: /home/shared/datasets/mesh_vpr/synt_melbourne/
Layout:
    database/                  — subdirectories organised by latitude
    database_images_paths.txt  — 394,632 relative paths under database/
    queries/                   — 1,249 flat .jpg files

Both DB and query filenames contain @utm_easting@utm_northing@ fields.
UTM is parsed from the full absolute path string: path.split("@")[1:3].
This works because the directory portions before the filename contain no
'@' character, so index 1 and 2 always land on easting and northing.

Ground truth: NearestNeighbors radius search at 25 m (same threshold as the
original MeshVPR paper and the SF-XL benchmark).
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

MESHVPR_ROOT = os.environ.get(
    "MESHVPR_PATH",
    "/home/shared/datasets/mesh_vpr/synt_melbourne/",
)
THRESHOLD_M = 25.0


class MeshVPRDataset(Dataset):
    """MeshVPR synthetic Melbourne dataset for VPR evaluation.

    DB paths are read from database_images_paths.txt (relative to database/).
    Query paths are listed from the queries/ flat directory.
    Ground truth uses 25 m UTM NearestNeighbors (matches paper threshold).

    Args:
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, input_transform=None):
        root = Path(MESHVPR_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"MeshVPR dataset not found at {MESHVPR_ROOT}. "
                "Set MESHVPR_PATH env var to the dataset root."
            )
        db_root = root / "database"
        q_dir = root / "queries"
        db_txt = root / "database_images_paths.txt"
        for p in (db_root, q_dir, db_txt):
            if not p.exists():
                raise FileNotFoundError(f"Expected path not found: {p}")

        self.input_transform = input_transform

        # DB: read relative paths from txt, build absolute paths.
        rel_paths = [
            p for p in db_txt.read_text().splitlines() if p.strip()
        ]
        self.dbImages = [str(db_root / p) for p in rel_paths]

        q_files = sorted(os.listdir(q_dir))
        self.qImages = [str(q_dir / f) for f in q_files]

        self.images = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        # Parse UTM from full absolute path string.
        # Directory parts before the filename contain no '@', so split("@")[1:3]
        # reliably gives easting and northing for both DB and query paths.
        db_utms = np.array(
            [(p.split("@")[1], p.split("@")[2]) for p in self.dbImages],
            dtype=float,
        )
        q_utms = np.array(
            [(p.split("@")[1], p.split("@")[2]) for p in self.qImages],
            dtype=float,
        )

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(db_utms)
        indices = knn.radius_neighbors(
            q_utms, radius=THRESHOLD_M, return_distance=False
        )
        self.ground_truth = np.empty(len(q_utms), dtype=object)
        for i, idx in enumerate(indices):
            self.ground_truth[i] = idx

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

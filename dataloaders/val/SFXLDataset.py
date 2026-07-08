import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import BallTree

SFXL_ROOT = os.environ.get("SFXL_PATH", "/home/shared/datasets/SF-XL/processed/test/")
THRESHOLD_M = 25.0
QUERY_SUBSETS = {"v1", "v2", "night", "occlusion"}


def _parse_utm(filename: str) -> tuple:
    """Return (easting, northing) from an SF-XL @-encoded filename.

    Format: @utm_e@utm_n@zone@hemi@lat@lon@key@...
    Splits basename on '@'; indices 1 and 2 carry easting and northing.
    """
    parts = Path(filename).name.split("@")
    return float(parts[1]), float(parts[2])


def _build_gt_utm(
    e_q: np.ndarray, n_q: np.ndarray,
    e_db: np.ndarray, n_db: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    """Return object array where gt[i] = DB index array within threshold_m metres.

    Euclidean BallTree on raw UTM coords is valid: all SF-XL test images share
    UTM zone 10S so planar distance equals metric distance.
    """
    tree = BallTree(np.column_stack([e_db, n_db]), metric="euclidean")
    indices = tree.query_radius(np.column_stack([e_q, n_q]), r=threshold_m)
    gt = np.empty(len(e_q), dtype=object)
    for i, idx in enumerate(indices):
        gt[i] = idx
    return gt


class SFXLDataset(Dataset):
    """SF-XL test dataset for VPR evaluation.

    Reads image paths from pre-built .txt path-list files and constructs ground
    truth from UTM coordinates encoded in filenames (25 m BallTree threshold).

    Args:
        query_subset: One of "v1", "v2", "night", "occlusion".
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, query_subset: str, input_transform=None):
        if query_subset not in QUERY_SUBSETS:
            raise ValueError(
                f"query_subset must be one of {QUERY_SUBSETS}, got '{query_subset}'"
            )
        root = Path(SFXL_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"SF-XL test root not found at {SFXL_ROOT}. "
                "Set SFXL_PATH env var to the dataset root."
            )
        db_txt = root / "database_images_paths.txt"
        q_txt = root / f"queries_{query_subset}_images_paths.txt"
        db_root = root / "database"
        q_root = root / f"queries_{query_subset}"
        for p in (db_txt, q_txt, db_root, q_root):
            if not p.exists():
                raise FileNotFoundError(f"Expected path not found: {p}")

        self.input_transform = input_transform
        db_rel = [p for p in db_txt.read_text().splitlines() if p.strip()]
        q_rel = [p for p in q_txt.read_text().splitlines() if p.strip()]

        self.dbImages = [str(db_root / p) for p in db_rel]
        self.qImages = [str(q_root / p) for p in q_rel]
        self.images = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        e_db = np.array([_parse_utm(p)[0] for p in db_rel])
        n_db = np.array([_parse_utm(p)[1] for p in db_rel])
        e_q = np.array([_parse_utm(p)[0] for p in q_rel])
        n_q = np.array([_parse_utm(p)[1] for p in q_rel])
        self.ground_truth = _build_gt_utm(e_q, n_q, e_db, n_db, THRESHOLD_M)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

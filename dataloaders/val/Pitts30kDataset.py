import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

PITTS_ROOT = os.environ.get("PITTS_PATH", "/home/eng/giborda/delavpr/datasets/pitts30k/")
THRESHOLD_M = 25.0
_SPLIT_MAP = {
    "pitts30k_test": "test",
    "pitts30k_val": "val",
}


def _parse_gps(filename: str) -> tuple:
    """Return (lat, lon) floats parsed from @utm_e@utm_n@...@lat@lon@... name."""
    parts = filename.split("@")
    return float(parts[5]), float(parts[6])


def _haversine_matrix(lat_q: np.ndarray, lon_q: np.ndarray,
                      lat_db: np.ndarray, lon_db: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance matrix (meters), shape (Q, DB)."""
    R = 6_371_000.0
    phi_q = np.radians(lat_q)[:, None]       # (Q, 1)
    phi_db = np.radians(lat_db)[None, :]     # (1, DB)
    dphi = phi_db - phi_q
    dlam = np.radians(lon_db)[None, :] - np.radians(lon_q)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi_q) * np.cos(phi_db) * np.sin(dlam / 2) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _build_gt(lat_q, lon_q, lat_db, lon_db, threshold_m: float) -> np.ndarray:
    """Return object array of index arrays; gt[i] = DB indices within threshold."""
    dist = _haversine_matrix(lat_q, lon_q, lat_db, lon_db)
    gt = np.empty(len(lat_q), dtype=object)
    for i in range(len(lat_q)):
        gt[i] = np.where(dist[i] < threshold_m)[0]
    return gt


class Pitts30kDataset(Dataset):
    """Pittsburgh 30k val/test loader for GPS-encoded flat-directory format.

    Args:
        which_ds: One of ``pitts30k_test`` or ``pitts30k_val``.
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, which_ds: str = "pitts30k_test", input_transform=None):
        assert which_ds in _SPLIT_MAP, f"Unknown split '{which_ds}'"
        split = _SPLIT_MAP[which_ds]
        root = Path(PITTS_ROOT) / split

        if not root.exists():
            raise FileNotFoundError(f"Pittsburgh split not found: {root}")

        self.input_transform = input_transform
        self.db_dir = root / "database"
        self.q_dir = root / "queries"

        db_files = sorted(os.listdir(self.db_dir))
        q_files = sorted(os.listdir(self.q_dir))

        self.dbImages = [str(self.db_dir / f) for f in db_files]
        self.qImages = [str(self.q_dir / f) for f in q_files]
        self.images = self.dbImages + self.qImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        lat_db = np.array([_parse_gps(f)[0] for f in db_files])
        lon_db = np.array([_parse_gps(f)[1] for f in db_files])
        lat_q = np.array([_parse_gps(f)[0] for f in q_files])
        lon_q = np.array([_parse_gps(f)[1] for f in q_files])
        self.ground_truth = _build_gt(lat_q, lon_q, lat_db, lon_db, THRESHOLD_M)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

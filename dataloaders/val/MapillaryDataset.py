import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# Expected layout:
#   $MSLS_VAL_PATH/database/   18,871 GPS-encoded images
#   $MSLS_VAL_PATH/query/      740 GPS-encoded images
# Filename format: @easting@northing@zone@hemi@lat@lon@key@...@.jpg
MSLS_VAL_PATH = os.environ.get("MSLS_VAL_PATH", "/home/shared/datasets/msls/val/")
THRESHOLD_M   = 25.0


def _parse_gps(filename: str) -> tuple:
    """Return (lat, lon) floats from MSLS GPS-encoded filename.

    Format: @utm_e@utm_n@zone@hemi@lat@lon@key@...@.jpg
    """
    parts = Path(filename).name.split("@")
    return float(parts[5]), float(parts[6])


def _haversine_matrix(lat_q: np.ndarray, lon_q: np.ndarray,
                      lat_db: np.ndarray, lon_db: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance matrix in metres, shape (Q, DB)."""
    R = 6_371_000.0
    phi_q  = np.radians(lat_q)[:, None]
    phi_db = np.radians(lat_db)[None, :]
    dphi   = phi_db - phi_q
    dlam   = np.radians(lon_db)[None, :] - np.radians(lon_q)[:, None]
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi_q) * np.cos(phi_db) * np.sin(dlam / 2) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _build_gt(lat_q, lon_q, lat_db, lon_db, threshold_m: float) -> np.ndarray:
    """Return object array gt where gt[i] = DB indices within threshold_m."""
    dist = _haversine_matrix(lat_q, lon_q, lat_db, lon_db)
    gt = np.empty(len(lat_q), dtype=object)
    for i in range(len(lat_q)):
        gt[i] = np.where(dist[i] < threshold_m)[0]
    return gt


class MSLS(Dataset):
    """MSLS validation set loader for GPS-encoded flat-directory format.

    Ground truth is built on-the-fly via haversine distance (25 m threshold)
    using GPS coordinates encoded in MSLS filenames.

    Args:
        input_transform: Optional torchvision transform pipeline.
        query_dir: Query subdirectory under MSLS_VAL_PATH. Defaults to
            'query'. Pass 'query_blur' or 'query_snow' for degraded subsets.
    """

    def __init__(self, input_transform=None, query_dir: str = "query"):
        root = Path(MSLS_VAL_PATH)
        if not root.exists():
            raise FileNotFoundError(
                f"MSLS val dataset not found at {MSLS_VAL_PATH}. "
                "Set MSLS_VAL_PATH env var to the dataset root."
            )
        db_dir = root / "database"
        q_dir  = root / query_dir
        for d in (db_dir, q_dir):
            if not d.exists():
                raise FileNotFoundError(
                    f"Expected directory '{d.name}' inside {MSLS_VAL_PATH}"
                )

        self.input_transform = input_transform

        db_files = sorted(os.listdir(db_dir))
        q_files  = sorted(os.listdir(q_dir))

        self.dbImages = [str(db_dir / f) for f in db_files]
        self.qImages  = [str(q_dir  / f) for f in q_files]
        self.images   = self.dbImages + self.qImages

        self.num_references = len(self.dbImages)
        self.num_queries    = len(self.qImages)

        lat_db = np.array([_parse_gps(f)[0] for f in db_files])
        lon_db = np.array([_parse_gps(f)[1] for f in db_files])
        lat_q  = np.array([_parse_gps(f)[0] for f in q_files])
        lon_q  = np.array([_parse_gps(f)[1] for f in q_files])
        self.ground_truth = _build_gt(lat_q, lon_q, lat_db, lon_db, THRESHOLD_M)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

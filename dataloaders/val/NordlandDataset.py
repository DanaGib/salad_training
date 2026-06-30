import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset: https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# Maintained by VPR-Bench: https://github.com/MubarizZaffar/VPR-Bench
# Expected layout:
#   $NORDLAND_PATH/database/          image files
#   $NORDLAND_PATH/queries/           image files
#   $NORDLAND_PATH/database_images_paths.txt   one filename per line
#   $NORDLAND_PATH/queries_images_paths.txt    one filename per line
NORDLAND_PATH = os.environ.get("NORDLAND_PATH", "/home/shared/datasets/nordland/test/")


class NordlandDataset(Dataset):
    """Nordland sequential traversal dataset for VPR evaluation.

    Nordland is a strictly ordered dataset where query frame i corresponds to
    database frame i.  Following the paper's evaluation protocol, a retrieval
    is considered correct when the returned database frame falls within ±2
    frames of the query index, i.e. gt[i] = [i-2, i-1, i, i+1, i+2]
    (clamped to valid indices).

    Args:
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, input_transform=None):
        root = Path(NORDLAND_PATH)
        if not root.exists():
            raise FileNotFoundError(
                f"Nordland dataset not found at {NORDLAND_PATH}. "
                "Set NORDLAND_PATH env var to the dataset root."
            )
        for subdir in ("database", "queries"):
            if not (root / subdir).exists():
                raise FileNotFoundError(
                    f"Expected '{subdir}' subdirectory inside {NORDLAND_PATH}"
                )

        self.input_transform = input_transform

        db_txt = root / "database_images_paths.txt"
        q_txt  = root / "queries_images_paths.txt"
        for p in (db_txt, q_txt):
            if not p.exists():
                raise FileNotFoundError(f"Image list not found: {p}")

        db_names = db_txt.read_text().splitlines()
        q_names  = q_txt.read_text().splitlines()

        self.dbImages = [str(root / "database" / f) for f in db_names]
        self.qImages  = [str(root / "queries"  / f) for f in q_names]
        self.images   = self.dbImages + self.qImages

        self.num_references = len(self.dbImages)
        self.num_queries    = len(self.qImages)

        # GT: query i matches any database frame within ±2 frames (paper threshold).
        self.ground_truth = np.array(
            [np.arange(max(0, i - 2), min(self.num_references, i + 3))
             for i in range(self.num_queries)],
            dtype=object,
        )

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

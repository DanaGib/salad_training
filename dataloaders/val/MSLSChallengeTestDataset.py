import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from dataloaders.val.msls_challenge_gt import CITIES, read_keys, read_utm, city_gt

MSLS_PATH = os.environ.get("MSLS_PATH", "/home/shared/datasets/msls_challenge/")
GT_META_ROOT = os.environ.get(
    "MSLS_CHALLENGE_GT_PATH",
    "/home/eng/giborda/delavpr/datasets/msls_challenge_GT/test_meta/",
)


class MSLSChallengeTest(Dataset):
    """MSLS Challenge test set with locally provided ground truth.

    Reads test_meta CSVs for 6 cities, builds GT via 25 m UTM BallTree (per
    city), and concatenates all cities into a single flat dataset. All DB
    images are placed first in self.images; all query images follow.

    Args:
        input_transform: Optional torchvision transform pipeline.
    """

    def __init__(self, input_transform=None):
        self.input_transform = input_transform
        msls_root = Path(MSLS_PATH)
        gt_root = Path(GT_META_ROOT)
        db_images, q_images, gt_list = [], [], []
        db_offset = 0

        for city in CITIES:
            city_meta = gt_root / city
            city_img = msls_root / "test" / city

            db_keys = read_keys(city_meta / "database")
            q_keys = read_keys(city_meta / "query")
            e_db, n_db = read_utm(city_meta / "database", db_keys)
            e_q, n_q = read_utm(city_meta / "query", q_keys)

            gt_list.extend(city_gt(e_q, n_q, e_db, n_db, db_offset))
            db_images += [
                str(city_img / "database" / "images" / f"{k}.jpg") for k in db_keys
            ]
            q_images += [
                str(city_img / "query" / "images" / f"{k}.jpg") for k in q_keys
            ]
            db_offset += len(db_keys)

        self.dbImages = db_images
        self.qImages = q_images
        self.images = db_images + q_images
        self.num_references = len(db_images)
        self.num_queries = len(q_images)
        self.ground_truth = np.empty(len(gt_list), dtype=object)
        for i, idx in enumerate(gt_list):
            self.ground_truth[i] = idx

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.images)

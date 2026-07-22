"""Evaluate cached VPR descriptors via FAISS — no model loading needed."""
import csv
import sys
import logging
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval import get_val_dataset
from utils.validation import get_validation_recalls

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DESC_CACHE = Path("logs/desc_cache")
CSV_OUT = Path("logs/eval/new_evals.csv")
IMAGE_SIZE = (322, 322)

DISPLAY_NAMES = {
    "SVOX_robotcar_sun": "svox_sun",    "SVOX_robotcar_snow": "svox_snow",
    "SVOX_robotcar_rain": "svox_rain",  "SVOX_robotcar_night": "svox_night",
    "SVOX_robotcar_overcast": "svox_overcast",
}

GROUP_A = [
    "pitts30k_val", "Nordland", "MSLS_blur", "MSLS_weather",
    "MSLS_Challenge_Test", "SFXL_v1", "SFXL_v2", "SFXL_night", "SFXL_occlusion",
    "SVOX", "SVOX_robotcar_sun", "SVOX_robotcar_snow",
    "SVOX_robotcar_rain", "SVOX_robotcar_night", "SVOX_robotcar_overcast",
]
GROUP_B = [
    "SVOX", "SVOX_robotcar_sun", "SVOX_robotcar_snow",
    "SVOX_robotcar_rain", "SVOX_robotcar_night", "SVOX_robotcar_overcast",
]

MODELS = [
    ("20260609_001439_epoch03",               GROUP_A),
    ("20260609_001439_last",                  GROUP_A),
    ("baseline_bs60_ep4",                     GROUP_A),
    ("baseline_bs80_ep6_es",                  GROUP_A),
    ("baseline_bs80_ep6_no_es",               GROUP_A),
    ("global_depth_ag0.5_noproj_ext",         GROUP_A),
    ("global_local_cos_none_ag0.1_al0.1_ext", GROUP_A),
    ("v2_global_local_ag0.02_al0.05",         GROUP_B),
    ("v2_global_local_ag0.05_al0.1",          GROUP_B),
]


def _append_csv(path: Path, row: dict) -> None:
    """Append one result row; write header only when creating the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def evaluate_from_cache(run_name: str, datasets: list) -> None:
    """Load cached .npy descriptors, run FAISS Recall@K, write result rows."""
    for ds_name in datasets:
        cache_dir = DESC_CACHE / run_name / ds_name
        db_path = cache_dir / "database_descriptors.npy"
        q_path  = cache_dir / "queries_descriptors.npy"
        if not (db_path.exists() and q_path.exists()):
            log.warning("Missing cache: %s / %s — skipping.", run_name, ds_name)
            continue
        try:
            _, _, _, gt = get_val_dataset(ds_name, image_size=IMAGE_SIZE)
        except FileNotFoundError as exc:
            log.warning("Dataset unavailable: %s (%s) — skipping.", ds_name, exc)
            continue
        db = torch.from_numpy(np.load(db_path).astype("float32"))
        q  = torch.from_numpy(np.load(q_path).astype("float32"))
        preds = get_validation_recalls(
            r_list=db, q_list=q, k_values=[1, 5, 10, 20],
            gt=gt, print_results=True, dataset_name=ds_name, faiss_gpu=False,
        )
        display = DISPLAY_NAMES.get(ds_name, ds_name)
        _append_csv(CSV_OUT, {
            "run_name": run_name, "dataset": display,
            "image_size": str(IMAGE_SIZE),
            "R@1":  round(preds[1]  * 100, 2),
            "R@5":  round(preds[5]  * 100, 2),
            "R@10": round(preds[10] * 100, 2),
            "R@20": round(preds[20] * 100, 2),
        })
        log.info("Done  %-45s  R@1 = %.2f", f"{run_name}/{display}", preds[1] * 100)


if __name__ == "__main__":
    for run_name, datasets in MODELS:
        evaluate_from_cache(run_name, datasets)
    print(f"\nResults written to {CSV_OUT}")

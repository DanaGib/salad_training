"""Descriptor extraction with disk and in-memory caching for VPR evaluation."""
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _try_load_from_disk(cache_dir: Path, dataset) -> Optional[Tuple]:
    """Return (db_desc, q_desc, all_descriptors) from cache_dir, or None.

    Validates array sizes against dataset.num_references and num_queries.
    """
    db_path, q_path = cache_dir / "database_descriptors.npy", cache_dir / "queries_descriptors.npy"
    if not (db_path.exists() and q_path.exists()):
        return None
    db, q = np.load(db_path), np.load(q_path)
    if len(db) != dataset.num_references or len(q) != dataset.num_queries:
        logger.warning(
            "Cache size mismatch in %s (DB %d vs %d, Q %d vs %d); re-extracting.",
            cache_dir, len(db), dataset.num_references, len(q), dataset.num_queries,
        )
        return None
    logger.info("Loaded descriptors from disk: %s", cache_dir)
    return db, q, np.vstack([db, q])


def extract_descriptors(
    model: torch.nn.Module,
    device: torch.device,
    dataset,
    args: argparse.Namespace,
    cache_dir: Path,
    db_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load or extract DB and query descriptors with two-level caching.

    Checks disk cache first, then an in-memory DB cache keyed by num_references
    (so datasets sharing the same DB, e.g. MSLS/MSLS_blur, skip DB inference on
    subsequent calls), then runs inference as a fallback.

    Args:
        model: VPRModel in eval mode.
        device: Inference device.
        dataset: Val dataset; __getitem__ returns (image_tensor, index).
        args: Argparse namespace — must expose batch_size, num_workers, save_descriptors.
        cache_dir: Per-run/per-dataset path for saving/loading .npy files.
        db_cache: Shared dict passed across dataset iterations for DB reuse.

    Returns:
        (db_desc, q_desc, all_descriptors, updated_db_cache) as numpy arrays.
    """
    if db_cache is None:
        db_cache = {}

    hit = _try_load_from_disk(cache_dir, dataset)
    if hit is not None:
        return hit[0], hit[1], hit[2], db_cache

    num_db, num_q = dataset.num_references, dataset.num_queries

    def _infer(indices: list, label: str) -> np.ndarray:
        """Run model inference over a subset and return stacked numpy array."""
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            shuffle=False,
        )
        return np.vstack([model(imgs.to(device)).cpu().numpy()
                          for imgs, _ in tqdm(loader, desc=label)])

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            if num_db in db_cache:
                logger.info("Reusing in-memory DB cache (%d images).", num_db)
                db_desc = db_cache[num_db]
            else:
                db_desc = _infer(list(range(num_db)), "DB descriptors")
                db_cache[num_db] = db_desc
            q_desc = _infer(list(range(num_db, num_db + num_q)), "Query descriptors")

    all_descriptors = np.vstack([db_desc, q_desc])

    if getattr(args, "save_descriptors", False):
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "database_descriptors.npy", db_desc)
        np.save(cache_dir / "queries_descriptors.npy", q_desc)
        logger.info("Saved descriptors to %s", cache_dir)

    return db_desc, q_desc, all_descriptors, db_cache

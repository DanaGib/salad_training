"""Unit tests for the AerialVL VPR dataset loader and ground-truth helpers.

All tests run on CPU against tiny synthetic images/directories built in a
pytest tmp_path fixture — no dependency on the real (75 GB) AerialVL
dataset and no GPU required.

Run from the repo root:
    pytest tests/test_aerialvl_dataset.py -v
"""
import importlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '.')

EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# 1. Filename parsing
# ---------------------------------------------------------------------------
def test_parse_query_lonlat():
    from utils.aerialvl_gt import parse_query_lonlat
    lon, lat = parse_query_lonlat("@120.425@36.596@.png")
    assert lon == 120.425
    assert lat == 36.596


def test_parse_db_tile_center():
    from utils.aerialvl_gt import parse_db_tile_center
    lon, lat = parse_db_tile_center("@map@120.40@36.50@120.50@36.60@.png")
    assert lon == 120.45
    assert lat == 36.55


# ---------------------------------------------------------------------------
# 2. Haversine ground-truth builder
# ---------------------------------------------------------------------------
def _meters_to_deg_lat(m: float) -> float:
    return m / 111_320.0


def test_build_gt_haversine_matches_within_threshold():
    from utils.aerialvl_gt import build_gt_haversine

    db_lonlat = np.array([[120.42, 36.60]])
    # ~30 m north of the DB point -> within a 50 m threshold.
    near_lat = 36.60 + _meters_to_deg_lat(30.0)
    # ~200 m north of the DB point -> outside a 50 m threshold.
    far_lat = 36.60 + _meters_to_deg_lat(200.0)
    q_lonlat = np.array([[120.42, near_lat], [120.42, far_lat]])

    gt = build_gt_haversine(q_lonlat, db_lonlat, threshold_m=50.0)

    assert list(gt[0]) == [0], "near query should match the single DB point"
    assert list(gt[1]) == [], "far query should have no matches"


def test_build_gt_haversine_output_shape():
    from utils.aerialvl_gt import build_gt_haversine

    db_lonlat = np.array([[120.40, 36.50], [120.50, 36.60], [120.60, 36.70]])
    q_lonlat = np.array([[120.40, 36.50], [120.55, 36.65]])

    gt = build_gt_haversine(q_lonlat, db_lonlat, threshold_m=50.0)

    assert gt.shape == (2,)
    assert gt.dtype == object
    assert list(gt[0]) == [0]


# ---------------------------------------------------------------------------
# 3. AerialVLDataset construction against a synthetic mini folder tree
# ---------------------------------------------------------------------------
def _make_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(path)


def _build_synthetic_dataset(tmp_path):
    root = tmp_path / "VPR"

    db_dir = root / "map_database" / "level_3"
    _make_png(db_dir / "@map@120.40@36.50@120.42@36.52@.png")
    _make_png(db_dir / "@map@120.50@36.60@120.52@36.62@.png")

    q1_dir = root / "query_images" / "query_images_1"
    _make_png(q1_dir / "@120.41@36.51@.png")
    _make_png(q1_dir / "@120.60@36.70@.png")

    q2_dir = root / "query_images" / "query_images_2"
    _make_png(q2_dir / "@120.51@36.61@.png")

    return root


def _reload_aerialvl_module():
    import dataloaders.val.AerialVLDataset as mod
    return importlib.reload(mod)


def test_aerialvl_dataset_counts_and_gt_shape(tmp_path, monkeypatch):
    root = _build_synthetic_dataset(tmp_path)
    monkeypatch.setenv("AERIALVL_PATH", str(root))
    monkeypatch.setenv("AERIALVL_LEVEL", "level_3")

    mod = _reload_aerialvl_module()
    ds = mod.AerialVLDataset(input_transform=None)

    assert ds.num_references == 2
    assert ds.num_queries == 3
    assert len(ds) == 5
    assert ds.ground_truth.shape == (3,)
    # Query 1 (@120.41@36.51@) sits inside DB tile 1's bbox -> near its
    # center, well within the 50 m threshold.
    assert 0 in ds.ground_truth[0]
    # Query 2 (@120.60@36.70@) is far from both DB tiles -> no match.
    assert list(ds.ground_truth[1]) == []


def test_aerialvl_dataset_getitem_returns_image(tmp_path, monkeypatch):
    root = _build_synthetic_dataset(tmp_path)
    monkeypatch.setenv("AERIALVL_PATH", str(root))
    monkeypatch.setenv("AERIALVL_LEVEL", "level_3")

    mod = _reload_aerialvl_module()
    ds = mod.AerialVLDataset(input_transform=None)

    img, idx = ds[0]
    assert idx == 0
    assert img.size == (4, 4)


def test_aerialvl_dataset_missing_root_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AERIALVL_PATH", str(tmp_path / "does_not_exist"))
    mod = _reload_aerialvl_module()

    try:
        mod.AerialVLDataset(input_transform=None)
        assert False, "expected FileNotFoundError for missing dataset root"
    except FileNotFoundError:
        pass

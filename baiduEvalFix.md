# Baidu Mall VPR Evaluation Fix

## Summary

Recall metrics on Baidu Mall were lower than published benchmarks due to a
single hardcoded distance threshold that was mismatched to the benchmark
protocol, plus a complete absence of startup logging that made the
misconfiguration silent.

---

## Root Cause Diagnosis

### Issue 1 — Wrong threshold in `BaiduDataset` (Berton / VPR-methods-evaluation split)

| | Before | After |
|---|---|---|
| Split | 5 207 DB / 3 208 Q (`database/` + `queries/`) | unchanged |
| Coord source | XY parsed from `@x@y@` filename | unchanged |
| Threshold | **10 m** (hardcoded) | **25 m** (default, overridable) |
| Benchmark | VPR-methods-evaluation / MegaLoc defines positives at **25 m** | matched |

Using 10 m instead of 25 m causes an artificial Recall@1 drop of ~10 %+ because
many true positives fall in the 10–25 m ring and are silently excluded from the
ground-truth set, penalising every correct retrieval in that band.

### Issue 2 — Threshold was hardcoded (no override path)

Both loaders used a bare module-level constant (`THRESHOLD_M = 10.0`).  There
was no env-var or constructor argument to reproduce a different threshold
without editing source, making cross-benchmark comparisons impossible from the
command line.

### Issue 3 — No startup logging

Neither loader printed the number of DB/query images or the active threshold.
A misconfiguration (wrong env-var path, wrong split folder, unexpected file
count) would pass silently through dataset construction and produce a mystery
Recall drop with no observable signal.

### Issue 4 — `BaiduOriginalDataset` threshold is correct

The official IDL_dataset_cvpr17 split (689 DB / 2 292 Q) with 10 m 3D Euclidean
matches AnyLoc / SegVLAD / VLAD-BuFF exactly.  No threshold change was made
here; parametrization and logging were added without changing the default.

---

## Files Changed

| File | Change |
|---|---|
| `dataloaders/val/BaiduDataset.py` | Default threshold 10 m → **25 m**; `threshold_m` constructor arg; `BAIDU_THRESHOLD_M` env-var; startup print |
| `dataloaders/val/baidu_original_dataloader.py` | `threshold_m` constructor arg; `BAIDU_ORIGINAL_THRESHOLD_M` env-var; startup print (default 10 m unchanged) |
| `eval.py` | `get_val_dataset()` accepts `baidu_threshold` + `baidu_original_threshold`; two new `--baidu_threshold` / `--baidu_original_threshold` CLI flags; `__main__` loop passes them through |

`utils/baidu_original_gt.py` was already correct — `build_gt_xyz` accepts
`threshold_m` as a plain argument and required no changes.

---

## Priority Order for Threshold Lookup

Both loaders resolve the threshold in this order (first non-None wins):

1. `threshold_m` constructor argument (highest priority)
2. `BAIDU_THRESHOLD_M` / `BAIDU_ORIGINAL_THRESHOLD_M` environment variable
3. Module default (25 m for Berton split, 10 m for original split)

---

## Benchmark Mode Reference

| Mode | Dataset key | Default threshold | Coord source | Expected split |
|---|---|---|---|---|
| VPR-methods-evaluation (Berton) | `baidu` | 25 m | `@x@y@` in filename | 5 207 DB / 3 208 Q |
| AnyLoc / SegVLAD / VLAD-BuFF | `baidu_original` | 10 m | `(x,y,z)` from `.camera` | 689 DB / 2 292 Q |

---

## Startup Log Format

Every evaluation run now prints a single line per dataset, e.g.:

```
[BaiduDataset]         split=Berton    db=5207  q=3208  threshold=25.0m  coord=XY-from-filename
[BaiduOriginalDataset] split=IDL_cvpr17  db=689   q=2292  threshold=10.0m  coord=XYZ-from-.camera
```

---

## CLI Usage Examples

```bash
# Standard VPR-methods-evaluation run (25 m, default)
python eval.py --ckpt_path model.ckpt --val_datasets baidu

# Reproduce AnyLoc numbers on Berton split (10 m override)
python eval.py --ckpt_path model.ckpt --val_datasets baidu --baidu_threshold 10.0

# Run original split at 25 m to match VPR-methods-evaluation on that split
python eval.py --ckpt_path model.ckpt --val_datasets baidu_original --baidu_original_threshold 25.0

# Override via env var (no CLI flag needed)
BAIDU_THRESHOLD_M=25 python eval.py --ckpt_path model.ckpt --val_datasets baidu
```

---

## Testing Plan

### Unit tests (pytest)

1. **Threshold resolution order** — construct `BaiduDataset(threshold_m=15.0)` and
   assert the log and GT use 15 m regardless of env var.
2. **Env-var override** — set `BAIDU_THRESHOLD_M=20` and construct without
   `threshold_m`; assert 20 m is used.
3. **Default fallback** — unset env var, no constructor arg; assert 25 m.
4. **GT size monotonicity** — for a small synthetic grid of XY points, assert
   `len(gt_positives_at_25m) >= len(gt_positives_at_10m)` for every query.
5. **Stem alignment** — `assert_stems_match` raises `ValueError` on a
   deliberately misaligned pair.
6. **`parse_cop` correctness** — write a temp `.camera` file with known CoP;
   assert returned array matches expected values.

### Integration / smoke test

```bash
python -c "
from dataloaders.val.BaiduDataset import BaiduDataset
ds = BaiduDataset()
assert ds.num_references > 0 and ds.num_queries > 0
print('BaiduDataset OK')
"
```

All tests should run green with no changes to model weights.

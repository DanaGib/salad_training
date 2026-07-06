# VPR Distillation Experiments V2

Reference for all depth-distillation training runs. Results append to `logs/eval/trials_v2.csv`.

---

## Shared Training Settings

| Parameter | V1 | V2 (224) | V2 (322) |
|---|---|---|---|
| Backbone | DINOv2 ViT-B/14, 4 trainable blocks | same | same |
| Optimizer | AdamW, lr=6e-5, weight_decay=9.5e-9 | same | same |
| LR schedule | linear warmdown, end_factor=0.2, total_iters=4000 | same | same |
| max_epochs | 4 | 6 | 6 |
| early_stop_patience | 0 (disabled) | 2, min_delta=0.0005 | 2, min_delta=0.0005 |
| batch_size | 60 | 80 | 40 |
| train img_size | 224×224 | 224×224 | 322×322 |
| eval img_size | 322×322 | 322×322 | 322×322 |
| val metric | pitts30k_val/recall_at_1 | same | same |
| Depth teacher | Depth-Anything-V2-Base-hf (frozen) | same | same |

---

## Loss Formulas

| Model type | Loss | Global | Local |
|---|---|:---:|:---:|
| salad_baseline | L = L_MS(d_salad) | - | - |
| salad_predictor_global | L = L_MS(d_salad) + α_g·L_MS(SALAD(MLP(f_bb))) + α_l·L_cos(MLP(f_bb), f_depth) | yes | yes |
| salad_global_depth | L = L_MS(d_salad) + α_g·L_MS(SALAD(f_depth)) | yes | - |
| salad_global_local_depth | L = L_MS(d_salad) + α_g·L_MS(SALAD(f_depth)) + α_l·L_cos(MLP(f_bb), f_depth) | yes | yes |

Notation: L_MS = MultiSimilarity loss, L_cos = cosine alignment, f_bb = backbone patch tokens, f_depth = depth teacher tokens.

---

## Architecture Notes

**Block A** (`salad_predictor_global`): backbone → SALAD → d_salad; backbone patches → AlignmentMLP (token_by_token, no norm) → d_pred tokens → SALAD → d_pred descriptor. Depth teacher is frozen and provides local cosine loss targets only; it does not feed the global SALAD branch.

**Block B** (`salad_global_depth`): backbone → SALAD → d_salad; depth teacher tokens (optionally through a learnable linear proj 768→768) → SALAD → d_depth. No AlignmentMLP; no local loss.

**Block C** (`salad_global_local_depth`): Block B global branch plus backbone patches → AlignmentMLP (token_by_token, no norm) → local cosine loss against depth teacher tokens.

**Block D-v2**: Same as Block C. Additionally, SALAD aggregator weights are loaded from the best baseline checkpoint before training (`model.init_aggregator_from`).

**Block E-v2**: Identical to Block C best config (ag=0.05, al=0.1, TBT) trained at 322×322, batch=40. Includes a 322-trained baseline for fair comparison.

---

## Baseline Reference

| run_name | pitts30k_test R@1 | amstertime R@1 | Nordland R@1 | MSLS R@1 |
|---|---|---|---|---|
| baseline_epoch03 | 92.65 | 58.65 | 55.85 | 92.16 |

---

## Block A — salad_predictor_global (global + local)

### V1 — train_trials.sh (batch=60, epochs=4)

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | p30k R@1 | ast R@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | pred_global_cos_none_ag0.05 | token_by_token | none | cosine | 0.05 | 0.2 | false | 224 | **92.27** | 58.25 |
| A2 | pred_global_cos_none_ag0.1  | token_by_token | none | cosine | 0.1  | 0.2 | false | 224 | 91.67 | 58.49 |
| A3 | pred_global_cos_none_ag0.5  | token_by_token | none | cosine | 0.5  | 0.2 | false | 224 | 92.12 | 56.13 |
| A4 | pred_global_cos_none_ag1.0  | token_by_token | none | cosine | 1.0  | 0.2 | false | 224 | 91.67 | 56.46 |

### V2 — train_v2.sh (batch=80, epochs=6) — finer α_g grid around A1

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | notes |
|---|---|---|---|---|---|---|---|---|---|
| A5 | v2_pred_global_ag0.05 | token_by_token | none | cosine | 0.05 | 0.2 | false | 224 | replicate A1 |
| A6 | v2_pred_global_ag0.02 | token_by_token | none | cosine | 0.02 | 0.2 | false | 224 | push lower |
| A7 | v2_pred_global_ag0.08 | token_by_token | none | cosine | 0.08 | 0.2 | false | 224 | push higher |

---

## Block B — salad_global_depth (global only)

### V1 — train_trials.sh (batch=60, epochs=4)

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | p30k R@1 | ast R@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | global_depth_ag0.05_noproj | none | none | mse | 0.05 | 0.0 | false | 224 | **92.25** | 57.68 |
| B2 | global_depth_ag0.1_noproj  | none | none | mse | 0.1  | 0.0 | false | 224 | 91.65 | 57.92 |
| B3 | global_depth_ag0.5_noproj  | none | none | mse | 0.5  | 0.0 | false | 224 | 91.67 | 58.08 |
| B4 | global_depth_ag1.0_noproj  | none | none | mse | 1.0  | 0.0 | false | 224 | 91.95 | 58.90 |
| B5 | global_depth_ag0.05_proj   | none | none | mse | 0.05 | 0.0 | true  | 224 | — | — |
| B6 | global_depth_ag0.1_proj    | none | none | mse | 0.1  | 0.0 | true  | 224 | — | — |
| B7 | global_depth_ag0.5_proj    | none | none | mse | 0.5  | 0.0 | true  | 224 | — | — |
| B8 | global_depth_ag1.0_proj    | none | none | mse | 1.0  | 0.0 | true  | 224 | — | — |

### V2 — train_v2.sh (batch=80, epochs=6) — extend α_g range, no proj

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | notes |
|---|---|---|---|---|---|---|---|---|---|
| B9  | v2_global_depth_ag0.05 | none | none | mse | 0.05 | 0.0 | false | 224 | replicate B1 |
| B10 | v2_global_depth_ag0.02 | none | none | mse | 0.02 | 0.0 | false | 224 | push lower |
| B11 | v2_global_depth_ag1.0  | none | none | mse | 1.0  | 0.0 | false | 224 | push higher |
| B12 | v2_global_depth_ag2.0  | none | none | mse | 2.0  | 0.0 | false | 224 | push higher |

---

## Block C — salad_global_local_depth (global + local)

### V1 — train_trials.sh (batch=60, epochs=4)

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | p30k R@1 | ast R@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | global_local_cos_none_ag0.05_al0.1 | token_by_token | none | cosine | 0.05 | 0.1 | false | 224 | **92.55** | **59.63** |
| C2 | global_local_cos_none_ag0.05_al0.5 | token_by_token | none | cosine | 0.05 | 0.5 | false | 224 | 92.33 | 56.62 |
| C3 | global_local_cos_none_ag0.1_al0.1  | token_by_token | none | cosine | 0.1  | 0.1 | false | 224 | 92.24 | 57.51 |
| C4 | global_local_cos_none_ag0.1_al0.5  | token_by_token | none | cosine | 0.1  | 0.5 | false | 224 | 92.00 | 58.25 |

Best v1 result: **C1** (ag=0.05, al=0.1) — 92.55 pitts30k / 59.63 amstertime.

### V2 — train_v2.sh (batch=80, epochs=6) — finer grid around C1

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | notes |
|---|---|---|---|---|---|---|---|---|---|
| C5 | v2_global_local_ag0.05_al0.1  | token_by_token | none | cosine | 0.05 | 0.10 | false | 224 | replicate C1 |
| C6 | v2_global_local_ag0.02_al0.1  | token_by_token | none | cosine | 0.02 | 0.10 | false | 224 | |
| C7 | v2_global_local_ag0.05_al0.05 | token_by_token | none | cosine | 0.05 | 0.05 | false | 224 | |
| C8 | v2_global_local_ag0.02_al0.05 | token_by_token | none | cosine | 0.02 | 0.05 | false | 224 | |
| C9 | v2_global_local_ag0.05_al0.2  | token_by_token | none | cosine | 0.05 | 0.20 | false | 224 | |

---

## Block D-v2 — Pretrained SALAD Init (salad_global_local_depth)

SALAD aggregator weights loaded from best baseline checkpoint before training (`model.init_aggregator_from`).

| ID | run_name | mlp_type | mlp_norm | loss_type | α_g | α_l | use_proj | img | batch | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | v2_trial4_init_ag0.05_al0.1  | token_by_token | none | cosine | 0.05 | 0.10 | false | 224 | 80 | pretrained SALAD init |
| D2 | v2_trial4_init_ag0.05_al0.05 | token_by_token | none | cosine | 0.05 | 0.05 | false | 224 | 80 | pretrained SALAD init |

---

## Block E-v2 — 322×322 Training (salad_baseline + salad_global_local_depth)

Directly addresses 224-train / 322-eval resolution mismatch.

| ID | run_name | model_type | mlp_type | loss_type | α_g | α_l | use_proj | img | batch | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | v2_baseline_322 | salad_baseline | none | — | 0.0 | 0.0 | false | 322 | 40 | 322-trained baseline |
| E2 | v2_global_local_322_ag0.05_al0.1 | salad_global_local_depth | token_by_token | cosine | 0.05 | 0.1 | false | 322 | 40 | mirrors C5 at 322 |

---

## Running Instructions

```bash
# Step 1 — Training sweep (all 16 v2 runs, ~8-10h)
tmux new -s train_v2
bash train_v2.sh /path/to/gsvcities
# Ctrl+B D to detach

# Step 2 — Re-evaluate existing v1 checkpoints on 4 benchmarks (~3-4h)
tmux new -s eval_ext
bash eval_extended.sh
# Ctrl+B D to detach

# Step 3 — Evaluate all v2 checkpoints (~3-4h, run next morning)
tmux new -s eval_v2
bash eval_v2.sh
# Ctrl+B D to detach
# Results written to logs/eval/trials_v2.csv
```

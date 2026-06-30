# Past Experiment Runs

All results are in logs/eval/ and W&B.
To re-run a single experiment:
  python main.py <overrides>  then  python eval.py --ckpt_path logs/checkpoints/<name>_*/last.ckpt --val_datasets pitts30k_test amstertime --image_size 322 322 --batch_size 256

## overnight suite (train_overnight.sh --suite)
Three experiments comparing normalization strategies on joint_depth.

| W&B run name              | model.type        | mlp.norm | loss   |
|---------------------------|-------------------|----------|--------|
| baseline                  | salad_baseline    | —        | —      |
| joint_depth_mse_after     | salad_joint_depth | after    | mse    |
| joint_depth_cosine_after  | salad_joint_depth | after    | cosine |

## alpha_sweep (train_alpha_sweep.sh)
Base: model.type=salad_joint_depth, mlp.norm=after, loss=mse

| W&B run name                   | alpha_local |
|--------------------------------|-------------|
| joint_depth_mse_after_alpha100 | 100         |
| joint_depth_mse_after_alpha500 | 500         |

## no_norm suite (train_no_norm_suite.sh)

| W&B run name            | model.type        | mlp.norm | loss   |
|-------------------------|-------------------|----------|--------|
| baseline                | salad_baseline    | —        | —      |
| joint_depth_mse_none    | salad_joint_depth | none     | mse    |
| joint_depth_cosine_none | salad_joint_depth | none     | cosine |

## linear_trials (train_linear_trials.sh)
Base: model.type=salad_joint_depth, model.mlp.type=linear

| W&B run name                    | mlp.norm | loss   | alpha_local |
|---------------------------------|----------|--------|-------------|
| baseline                        | —        | —      | default     |
| linear_depth_mse_none           | none     | mse    | default     |
| linear_depth_mse_after          | after    | mse    | default     |
| linear_depth_cosine_none        | none     | cosine | default     |
| linear_depth_cosine_after       | after    | cosine | default     |
| linear_depth_mse_none_alpha100  | none     | mse    | 100         |
| linear_depth_mse_after_alpha100 | after    | mse    | 100         |
| linear_depth_mse_none_alpha500  | none     | mse    | 500         |
| linear_depth_mse_after_alpha500 | after    | mse    | 500         |

## smoke test pattern (train_smoke_test.sh)
Run any single model type with custom alpha_global, then auto-eval:

    python main.py model.type=<type> loss.alpha_global=<ag> loss.alpha_local=0.2 \
      model.mlp.normalization=none loss.alignment_loss_type=cosine \
      training.val_set_names=[pitts30k_val] wandb.run_name=<name>

    python eval.py --ckpt_path logs/checkpoints/<name>_*/last.ckpt \
      --val_datasets pitts30k_test amstertime --image_size 322 322 --batch_size 256
import csv
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf

from vpr_model import VPRModel
from utils.extraction import extract_descriptors
from utils.validation import get_validation_recalls

VAL_DATASETS = [
    'MSLS', 'MSLS_Test',
    'MSLS_Challenge_Test',
    'MSLS_blur', 'MSLS_weather',
    'pitts30k_test', 'pitts30k_val',
    'pitts250k_test',
    'Nordland', 'SPED',
    'amstertime',
    'SFXL_v1', 'SFXL_v2', 'SFXL_night', 'SFXL_occlusion',
    'SVOX',
    'SVOX_robotcar_sun', 'SVOX_robotcar_snow', 'SVOX_robotcar_rain',
    'SVOX_robotcar_night', 'SVOX_robotcar_overcast',
]


def input_transform(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

def get_val_dataset(dataset_name, image_size=None):
    """Build the requested validation dataset, importing the module lazily.

    Args:
        dataset_name: One of the keys in VAL_DATASETS.
        image_size: Optional (H, W) tuple passed to input_transform.

    Returns:
        Tuple of (dataset, num_references, num_queries, ground_truth).
    """
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    if 'nordland' in dataset_name:
        from dataloaders.val.NordlandDataset import NordlandDataset
        ds = NordlandDataset(input_transform=transform)
    elif 'sfxl' in dataset_name:
        subset = dataset_name.split('_', 1)[1]   # 'v1', 'v2', 'night', 'occlusion'
        from dataloaders.val.SFXLDataset import SFXLDataset
        ds = SFXLDataset(query_subset=subset, input_transform=transform)
    elif dataset_name == 'msls_challenge_test':
        from dataloaders.val.MSLSChallengeTestDataset import MSLSChallengeTest
        ds = MSLSChallengeTest(input_transform=transform)
    elif dataset_name == 'msls_blur':
        from dataloaders.val.MapillaryDataset import MSLS
        ds = MSLS(input_transform=transform, query_dir='query_blur')
    elif dataset_name == 'msls_weather':
        from dataloaders.val.MapillaryDataset import MSLS
        ds = MSLS(input_transform=transform, query_dir='query_snow')
    elif 'msls_test' in dataset_name:
        from dataloaders.val.MapillaryTestDataset import MSLSTest
        ds = MSLSTest(input_transform=transform)
    elif 'msls' in dataset_name:
        from dataloaders.val.MapillaryDataset import MSLS
        ds = MSLS(input_transform=transform)
    elif dataset_name in ('pitts30k_test', 'pitts30k_val'):
        from dataloaders.val.Pitts30kDataset import Pitts30kDataset
        ds = Pitts30kDataset(which_ds=dataset_name, input_transform=transform)
    elif 'pitts' in dataset_name:
        from dataloaders.val.PittsburghDataset import PittsburghDataset
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)
    elif 'amstertime' in dataset_name:
        from dataloaders.val.AmsterTimeDataset import AmsterTimeDataset
        ds = AmsterTimeDataset(split='test', input_transform=transform)
    elif 'sped' in dataset_name:
        from dataloaders.val.SPEDDataset import SPEDDataset
        ds = SPEDDataset(input_transform=transform)
    elif 'svox_robotcar' in dataset_name:
        subset = dataset_name.split('svox_robotcar_', 1)[1]  # 'sun', 'snow', ...
        from dataloaders.val.RobotCarSVOXDataset import RobotCarSVOXDataset
        ds = RobotCarSVOXDataset(query_subset=subset, input_transform=transform)
    elif 'svox' in dataset_name:
        from dataloaders.val.SVOXDataset import SVOXDataset
        ds = SVOXDataset(input_transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)

def load_model(ckpt_path):
    """Load a VPRModel checkpoint for inference.

    Always initialises with salad_baseline architecture (no depth teacher).
    Works for both salad_baseline and salad_joint_depth checkpoints because
    depth_teacher.* weights are stripped from the state dict before loading.

    Args:
        ckpt_path: Path to a Lightning .ckpt file.

    Returns:
        VPRModel in eval mode on CUDA.
    """
    cfg = OmegaConf.create({
        "model": {
            "type": "salad_baseline",
            "backbone": {
                "arch": "dinov2_vitb14",
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            "aggregator": {
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        },
        "loss": {
            "vpr_loss": "MultiSimilarityLoss",
            "miner": "MultiSimilarityMiner",
            "miner_margin": 0.1,
        },
        "training": {"faiss_gpu": False, "log_interval": 1000},
    })

    model = VPRModel(cfg)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # Lightning checkpoints wrap weights under 'state_dict'; fall back to the
    # raw dict for plain torch.save() exports.
    sd = checkpoint.get('state_dict', checkpoint)
    # Strip training-only modules saved during joint-depth runs:
    # depth_teacher (frozen teacher) and alignment_mlp (distillation head)
    # are not needed for inference.
    skip = ('depth_teacher.', 'alignment_mlp.')
    sd = {k: v for k, v in sd.items() if not k.startswith(skip)}
    model.load_state_dict(sd, strict=True)
    model = model.eval().to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument("--ckpt_path", type=str, required=True, default=None, help="Path to the checkpoint")
    
    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size', nargs='*', default=None, help='Image size (int, tuple or None)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument(
        '--run_name', type=str, default=None,
        help='Human-readable run label written as the run_name column in the CSV. '
             'Defaults to the checkpoint filename stem.',
    )
    parser.add_argument(
        '--csv_name', type=str, default=None,
        help='Filename stem for the output CSV (written to logs/eval/<csv_name>.csv). '
             'Defaults to run_name. Set the same value across multiple eval calls '
             'to accumulate all results in one file.',
    )
    parser.add_argument(
        '--csv_path', type=str, default=None,
        help='Absolute or relative path for the output CSV. Overrides --csv_name '
             'and the default logs/eval/ directory when set.',
    )
    parser.add_argument(
        '--extra_params', type=str, default=None,
        help='JSON string of extra metadata columns merged into every CSV row, '
             'e.g. \'{"model_type":"salad_global_local_depth","alpha_global":0.05}\'. '
             'Columns appear between image_size and R@1 in the output.',
    )
    parser.add_argument(
        '--save_descriptors', action='store_true',
        help='Save extracted descriptors to disk under --desc_cache_dir for future reuse.',
    )
    parser.add_argument(
        '--desc_cache_dir', type=str, default='logs/desc_cache',
        help='Root folder for cached descriptor .npy files (organised by run_name/dataset).',
    )
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='Number of DataLoader workers for descriptor extraction.',
    )

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError('Invalid image size, must be int, tuple or None')
        
        args.image_size = tuple(map(int, args.image_size))

    return args


def save_results_csv(results: list, run_name: str, csv_path: Optional[Path] = None) -> Path:
    """Append eval results to a CSV file.

    When csv_path is given, results are written to that exact location.
    Otherwise the file is placed under logs/eval/ named after run_name so all
    evaluations for the same run accumulate in one place. The header is written
    only when the file is new.

    Args:
        results: List of dicts, one per dataset, with recall columns.
        run_name: Human-readable label used as the CSV filename stem when
            csv_path is not provided.
        csv_path: Optional explicit output file path; overrides run_name stem.

    Returns:
        Path to the written CSV file.
    """
    if csv_path is None:
        csv_dir = Path(__file__).parent / "logs" / "eval"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{run_name}.csv"
    else:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    return csv_path


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()
    extra_params = json.loads(args.extra_params) if args.extra_params else {}

    model = load_model(args.ckpt_path)
    results = []
    run_name = args.run_name or Path(args.ckpt_path).stem
    db_cache = {}

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        cache_dir = Path(args.desc_cache_dir) / run_name / val_name

        print(f'Evaluating on {val_name}')
        db_desc, q_desc, all_descriptors, db_cache = extract_descriptors(
            model, torch.device('cuda'), val_dataset, args, cache_dir, db_cache
        )

        r_list = torch.from_numpy(db_desc)
        q_list = torch.from_numpy(q_desc)

        print(f'Descriptor dimension {all_descriptors.shape[1]}')
        print('total_size', all_descriptors.shape[0], num_queries + num_references)

        testing = 'msls_test' in val_name.lower()

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 20],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
            testing=testing,
        )

        if testing:
            val_dataset.save_predictions(preds, args.ckpt_path + '.' + model.agg_arch + '.preds.txt')
        else:
            print(
                f"RECALLS {val_name}"
                f" R@1={preds[1]*100:.2f}"
                f" R@5={preds[5]*100:.2f}"
                f" R@10={preds[10]*100:.2f}"
                f" R@20={preds[20]*100:.2f}"
            )
            row = {
                "run_name": run_name,
                "checkpoint": Path(args.ckpt_path).name,
                "dataset": val_name,
                "image_size": str(args.image_size),
            }
            row.update(extra_params)
            row.update({
                "R@1":  round(preds[1]  * 100, 2),
                "R@5":  round(preds[5]  * 100, 2),
                "R@10": round(preds[10] * 100, 2),
                "R@20": round(preds[20] * 100, 2),
            })
            results.append(row)

        print('========> DONE!\n\n')

    if results:
        explicit_path = Path(args.csv_path) if args.csv_path else None
        csv_name = args.csv_name or run_name
        out_path = save_results_csv(results, csv_name, csv_path=explicit_path)
        print(f"Results saved to {out_path}")


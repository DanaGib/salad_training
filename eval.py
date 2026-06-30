import csv
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

from vpr_model import VPRModel
from utils.validation import get_validation_recalls

VAL_DATASETS = [
    'MSLS', 'MSLS_Test',
    'pitts30k_test', 'pitts30k_val',
    'pitts250k_test',
    'Nordland', 'SPED',
    'amstertime',
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
        "training": {"faiss_gpu": False},
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


def save_results_csv(results: list, ckpt_path: str) -> Path:
    """Write eval results to a timestamped CSV under logs/eval/.

    Args:
        results: List of dicts, one per dataset, with recall columns.
        ckpt_path: Path to the evaluated checkpoint (used to name the file).

    Returns:
        Path to the written CSV file.
    """
    csv_dir = Path(__file__).parent / "logs" / "eval"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"{Path(ckpt_path).stem}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return csv_path


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()

    model = load_model(args.ckpt_path)
    results = []

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        val_loader = DataLoader(val_dataset, num_workers=16, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')

        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        print('total_size', descriptors.shape[0], num_queries + num_references)

        testing = 'msls_test' in val_name.lower()

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
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
            results.append({
                "checkpoint": Path(args.ckpt_path).name,
                "dataset": val_name,
                "image_size": str(args.image_size),
                "R@1":  round(preds[1]  * 100, 2),
                "R@5":  round(preds[5]  * 100, 2),
                "R@10": round(preds[10] * 100, 2),
                # "R@15": round(preds[15] * 100, 2),
                "R@20": round(preds[20] * 100, 2),
                # "R@25": round(preds[25] * 100, 2),
            })

        del descriptors
        print('========> DONE!\n\n')

    if results:
        csv_path = save_results_csv(results, args.ckpt_path)
        print(f"Results saved to {csv_path}")


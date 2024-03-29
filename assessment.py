import torch
from torch.utils import data as torch_data
import torchvision
import torchvision.transforms as transforms

import numpy as np

from tqdm import tqdm
from utils import metrics, parsers, experiment_manager, networks, helpers
from utils.experiment_manager import CfgNode
from pathlib import Path
from timeit import default_timer as timer


def model_assessment_cifar10(cfg: CfgNode, train: bool = False, hyper_params: str = None):
    if hyper_params is not None:
        run_name = f'{cfg.NAME}_{hyper_params}'
        assert(len(hyper_params) == 6)
        cfg.MODEL.NUM_ESTIMATORS = int(hyper_params[1])
        cfg.MODEL.ALPHA = int(hyper_params[3])
        cfg.MODEL.GAMMA = int(hyper_params[5])
    else:
        run_name = f'{cfg.NAME}'

    runs = sorted((Path(cfg.PATHS.OUTPUT) / 'networks').glob(f'{run_name}_run_*'))
    if not runs:
        # We only have a single run
        runs = [Path(cfg.PATHS.OUTPUT) / 'networks' / f'{run_name}.pt']
        
    data_dict = {}
    for run_path in runs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = networks.load_checkpoint(cfg, device, net_file=run_path)
        net.to(device)
        net.eval()

        n_params = networks.count_parameters(net)
        print(f'n parameters: {n_params}')

        measurer = metrics.ClassificationMetrics(cfg.MODEL.OUT_CHANNELS, cfg.MODEL.ENSEMBLE)
        ood_measurer = metrics.OODMetrics(1, cfg.MODEL.ENSEMBLE)

        transform_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_svhn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])

        dataset_cifar10 = torchvision.datasets.CIFAR10(root=Path(cfg.PATHS.DATASET), train=train, download=True,
                                            transform=transform_cifar10)
        dataset_svhn = torchvision.datasets.SVHN(root=Path(cfg.PATHS.DATASET), split='train' if train else 'test',
                                                 download=True, transform=transform_svhn)

        dataloader_kwargs = {
            'batch_size': cfg.TRAINER.BATCH_SIZE,
            'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
            'shuffle': False,
        }

        dataloader = torch_data.DataLoader(dataset_cifar10, **dataloader_kwargs)
        svhn_dataloader = torch_data.DataLoader(dataset_svhn, **dataloader_kwargs)

        start = timer()
        for step, (images, labels) in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                logits = net(images.to(device)).cpu().detach()
            measurer.add_sample(logits, labels)
            ood_measurer.add_sample(logits, torch.ones_like(labels))
        end = timer()

        for step, (images, labels) in enumerate(tqdm(svhn_dataloader)):
            with torch.no_grad():
                logits = net(images.to(device)).cpu().detach()
            ood_measurer.add_sample(logits, torch.zeros_like(labels))

        data = {
            'acc': float(measurer.accuracy()),
            'nll': float(measurer.negative_log_likelihood()),
            'ece': float(measurer.calibration_error()),
            'auc': float(ood_measurer.auc()),
            'aupr': float(ood_measurer.aupr()),
            'fpr95': float(ood_measurer.fpr95()),
            'time': float(end - start),
            'images': len(dataset_cifar10),
            'params': int(n_params),
        }
        data_dict[run_path.stem] = data

        print(data)

        out_path = Path(cfg.PATHS.OUTPUT) / 'assessment' / f'{run_path.stem}.json'
        helpers.write_json(out_path, data)

    avg_data = {m:round(np.mean([d[m] for d in data_dict.values()]), 3) for m in data}
    print('Avg stats:', avg_data)
    std_data = {m: round(np.std([d[m] for d in data_dict.values()]), 3) for m in data}
    print('Std stats', std_data)

    out_data = {
        'mean': avg_data,
        'std': std_data,
        'runs': len(runs),
    }

    out_path = Path(cfg.PATHS.OUTPUT) / 'assessment' / f'{run_name}.json'
    helpers.write_json(out_path, out_data)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    model_assessment_cifar10(cfg, hyper_params=args.hyper_params)

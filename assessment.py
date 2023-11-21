import torch
from torch.utils import data as torch_data
import torchvision
import torchvision.transforms as transforms

import wandb
from tqdm import tqdm
from utils import metrics, parsers, experiment_manager, networks, helpers
from utils.experiment_manager import CfgNode
from pathlib import Path


def model_assessment_cifar10(cfg: CfgNode, train: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = networks.load_checkpoint(cfg, device)
    net.to(device)
    net.eval()

    n_params = networks.count_parameters(net)
    print(f'n parameters: {n_params}')

    measurer = metrics.ClassificationMetrics(cfg.MODEL.OUT_CHANNELS, cfg.MODEL.ENSEMBLE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(root=Path(cfg.PATHS.DATASET), train=train, download=True,
                                           transform=transform)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': False,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    for step, (images, labels) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            logits = net(images.to(device))
        measurer.add_sample(logits, labels)

    data = {
        'acc': float(measurer.accuracy()),
        'nll': float(measurer.negative_log_likelihood()),
        'ece': float(measurer.calibration_error()),
        'auc': float(measurer.auc()),
        'aupr': float(measurer.aupr()),
        'fpr95': float(measurer.fpr95()),
    }

    out_path = Path(cfg.PATHS.OUTPUT) / 'assessment' / f'{cfg.NAME}.json'
    helpers.write_json(out_path, data)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    model_assessment_cifar10(cfg)

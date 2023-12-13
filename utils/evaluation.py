import torch
from torch.utils import data as torch_data
import torchvision
import torchvision.transforms as transforms

import wandb
from tqdm import tqdm
from utils import metrics
from utils.experiment_manager import CfgNode
from pathlib import Path


def model_evaluation_cifar10(net, cfg: CfgNode, train: bool, epoch: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

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
        measurer.add_sample(logits.cpu().detach(), labels)

    # assessment
    acc = measurer.accuracy()
    run_type = 'Train' if train else 'Val'
    print(f'Accuracy {run_type} {acc:.2f}')
    wandb.log({
        f'{run_type} acc': acc,
        f'{run_type} nll': measurer.negative_log_likelihood(),
        f'{run_type} ece': measurer.calibration_error(),
        # f'{run_type} auc': measurer.auc(),
        # f'{run_type} aupr': measurer.aupr(),
        # f'{run_type} fpr95': measurer.fpr95(),
        'step': step,
        'epoch': epoch,
    })

    return acc


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import roc_curve

    # Assuming you have toy tensors for predictions and labels
    # Replace these with your actual toy tensors or generate random ones
    num_samples = 1000
    num_classes = 5

    # Random toy predictions (replace this with your actual tensor)
    all_predictions = torch.randn((num_samples, num_classes))

    # Random toy labels (replace this with your actual tensor)
    all_labels = torch.randint(0, num_classes, (num_samples,))

    # Convert labels to one-hot encoding
    one_hot_labels = F.one_hot(all_labels)

    # Calculate FPR and TPR using sklearn's roc_curve
    fpr, tpr, _ = roc_curve(one_hot_labels.ravel(), all_predictions.ravel())

    # Find the index where recall is closest to 95%
    recall_at_95 = 0.95
    recall_index = np.argmax(tpr >= recall_at_95)

    # Calculate FPR95
    fpr95 = fpr[recall_index]

    print(f"FPR at 95% Recall (FPR95): {fpr95:.4f}")
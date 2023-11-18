import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, experiment_manager, networks
from utils.experiment_manager import CfgNode
from scipy import stats
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics import CalibrationError, AUROC, AveragePrecision
from sklearn.metrics import roc_curve
import torch.nn as nn


class RegressionEvaluation(object):
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add_sample_numpy(self, pred: np.ndarray, label: np.ndarray):
        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def add_sample_torch(self, pred: torch.tensor, label: torch.tensor):
        pred = pred.float().detach().cpu().numpy()
        label = label.float().detach().cpu().numpy()
        self.add_sample_numpy(pred, label)

    def reset(self):
        self.predictions = []
        self.labels = []

    def root_mean_square_error(self) -> float:
        return np.sqrt(np.sum(np.square(np.array(self.predictions) - np.array(self.labels))) / len(self.labels))

    def r_square(self) -> float:
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.labels, self.predictions)
        return r_value


class ClassificationEvaluation(object):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.predictions = []
        self.labels = []
        self.total = 0
        self.correct = 0

        # Initialize the ECE
        self.ece = CalibrationError(task='multiclass', num_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.nll = 0
        self.auroc = AUROC(task='multiclass', num_classes=n_classes)
        self.average_precision = AveragePrecision(task='multiclass', num_classes=n_classes)
        self.probabilities = []

    def add_sample(self, logit: torch.tensor, label: torch.tensor):
        prob = F.softmax(logit, dim=1)
        # loss = F.nll_loss(prob, label)
        loss = self.criterion(logit, label)
        self.nll += loss.item()
        self.ece.update(prob, label)
        self.probabilities.append(prob)

        _, pred = torch.max(prob, dim=1)

        self.total += label.size(0)
        self.correct += pred.eq(label).sum().item()

        self.predictions.extend(pred.numpy().flatten())

        self.labels.extend(label.numpy().flatten())

    def reset(self):
        self.predictions = []
        self.labels = []

    def accuracy(self) -> float:
        # return np.mean(np.array(self.predictions) == np.array(self.labels)) * 100
        return 100. * self.correct / self.total

    def calibration_error(self) -> float:
        return self.ece.compute()

    def negative_log_likelihood(self) -> float:
        return self.nll

    def auc(self) -> float:
        return self.auroc(torch.cat(self.probabilities, dim=0), torch.Tensor(self.labels).long())

    def aupr(self) -> float:
        return self.average_precision(torch.cat(self.probabilities, dim=0), torch.Tensor(self.labels).long())

    def fpr95(self) -> float:
        # Random toy predictions (replace this with your actual tensor)
        all_predictions = torch.cat(self.probabilities, dim=0)

        # Random toy labels (replace this with your actual tensor)
        all_labels = torch.Tensor(self.labels).long()

        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(all_labels)

        # Calculate FPR and TPR using sklearn's roc_curve
        fpr, tpr, _ = roc_curve(one_hot_labels.ravel(), all_predictions.ravel())

        # Find the index where recall is closest to 95%
        recall_at_95 = 0.95
        recall_index = np.argmax(tpr >= recall_at_95)

        # Calculate FPR95
        return fpr[recall_index]


def model_evaluation_grid(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, run_type: str, epoch: float,
                          max_samples: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()
    dataset = datasets.GridPopulationDataset(cfg, run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    max_samples = len(dataset) if max_samples is None else max_samples
    counter = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            img = batch['x'].to(device)
            label = batch['y'].to(device)
            pred = net(img)
            measurer.add_sample_torch(pred, label)
            counter += 1
            if counter == max_samples:
                break

    # assessment
    rmse = measurer.root_mean_square_error()
    rsquare = measurer.r_square()
    print(f'RMSE {run_type} {rmse:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse,
        f'{run_type} r2': rsquare,
        'step': step,
        'epoch': epoch,
    })

    return rmse


def model_evaluation_cifar10(net, cfg: CfgNode, train: bool, epoch: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = ClassificationEvaluation(cfg.MODEL.OUT_CHANNELS)
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
        f'{run_type} auc': measurer.auc(),
        f'{run_type} aupr': measurer.aupr(),
        f'{run_type} fpr95': measurer.fpr95(),
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
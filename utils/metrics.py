import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import CalibrationError, AUROC, AveragePrecision

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class ClassificationMetrics(object):
    def __init__(self, n_classes: int, ensemble: bool = False):
        self.n_classes = n_classes
        self.ensemble = ensemble

        self.predictions = []
        self.labels = []
        self.total = 0
        self.correct = 0
        self.batch_n = 0

        # Initialize the ECE
        self.ece = CalibrationError(task='multiclass', num_classes=n_classes)
        self.criterion = torch.nn.NLLLoss()
        self.loss = 0
        self.probabilities = []

    def add_sample(self, logits: torch.tensor, labels: torch.tensor):
        if self.ensemble:
            sm = nn.LogSoftmax(dim=2)
            probs = sm(logits)
            probs = torch.mean(probs, dim=0)
        else:
            sm = nn.LogSoftmax(dim=1)
            probs = sm(logits)

        loss = self.criterion(probs, labels)
        self.loss += loss.item()
        self.batch_n += 1

        self.ece.update(probs, labels)

        _, preds = torch.max(probs, dim=1)
        self.total += labels.size(0)
        self.correct += preds.eq(labels).sum().item()

        self.probabilities.append(probs)
        self.predictions.append(preds)
        self.labels.append(labels)

    def accuracy(self) -> float:
        return 100. * self.correct / self.total

    def calibration_error(self) -> float:
        return self.ece.compute()

    def negative_log_likelihood(self) -> float:
        return self.loss / self.batch_n

class OODMetrics(object):
    def __init__(self, n_classes: int, ensemble: bool = False):
        self.n_classes = n_classes
        self.ensemble = ensemble

        self.labels = []
        self.total = 0
        self.correct = 0
        self.batch_n = 0

        # Initialize the ECE
        self.ece = CalibrationError(task='binary', num_classes=n_classes)
        self.criterion = torch.nn.NLLLoss()
        self.loss = 0
        self.probabilities = []

    def add_sample(self, logits: torch.tensor, labels: torch.tensor):
        if self.ensemble:
            sm = nn.Softmax(dim=2)
            logits = logits.cpu().detach()
            probs = sm(logits)
            probs = torch.mean(probs, dim=0)
        else:
            sm = nn.Softmax(dim=1)
            logits = logits.cpu().detach()
            probs = sm(logits)

        self.batch_n += 1

        max_prob, preds = torch.max(probs, dim=1)

        self.ece.update(max_prob, labels)

        self.total += labels.size(0)

        self.probabilities.append(max_prob)
        self.labels.append(labels)

    def auc(self) -> float:
        auroc = AUROC(task='binary', num_classes=self.n_classes)
        preds = torch.cat(self.probabilities, dim=0)
        target = torch.cat(self.labels)
        return auroc(preds, target) * 100

    def aupr(self) -> float:
        # Adapted from https://github.com/tayden/ood-metrics/blob/main/ood_metrics/metrics.py
        # Measuring AUPR Out
        preds = torch.cat(self.probabilities, dim=0).ravel()
        preds_out = -preds
        target = torch.cat(self.labels).ravel()
        target_out = 1 - target
        precision, recall, _ = precision_recall_curve(target_out, preds_out, pos_label=1)
        return auc(recall, precision) * 100

    def fpr95(self) -> float:
        # From https://github.com/tayden/ood-metrics/blob/main/ood_metrics/metrics.py
        preds = torch.cat(self.probabilities, dim=0).ravel()
        preds_out = -preds
        target = torch.cat(self.labels).ravel()
        target_out = 1 - target
        fpr, tpr, _ = roc_curve(target_out, preds_out)
        if all(tpr < 0.95):
            # No threshold allows TPR >= 0.95
            return 0
        elif all(tpr >= 0.95):
            # All thresholds allow TPR >= 0.95, so find lowest possible FPR
            idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
            return min(map(lambda idx: fpr[idx], idxs)) * 100
        else:
            # Linear interp between values to get FPR at TPR == 0.95
            return np.interp(0.95, tpr, fpr) * 100

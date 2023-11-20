import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import CalibrationError, AUROC, AveragePrecision

import numpy as np
from sklearn.metrics import roc_curve


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
        sm = nn.LogSoftmax(dim=1)

        if self.ensemble:
            logits = [l.cpu().detach() for l in logits]
            probs = [sm(l) for l in logits]
            probs = torch.mean(torch.stack(probs), dim=0)
        else:
            logits = logits.cpu().detach()
            probs = sm(logits)

        loss = self.criterion(probs, labels)
        self.loss += loss.item()
        self.batch_n += 1

        # probs = F.softmax(logits, dim=1)
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

    def auc(self) -> float:
        auroc = AUROC(task='multiclass', num_classes=self.n_classes)
        preds = torch.cat(self.probabilities, dim=0)
        target = torch.cat(self.labels)
        return auroc(preds, target)

    def aupr(self) -> float:
        average_precision = AveragePrecision(task='multiclass', num_classes=self.n_classes)
        preds = torch.cat(self.probabilities, dim=0)
        target = torch.cat(self.labels)
        return average_precision(preds, target)

    def fpr95(self) -> float:
        all_predictions = torch.cat(self.probabilities, dim=0)
        all_labels = torch.cat(self.labels)

        one_hot_labels = F.one_hot(all_labels)

        fpr, tpr, _ = roc_curve(one_hot_labels.ravel(), all_predictions.ravel())

        recall_index = np.argmax(tpr >= 0.95)

        return fpr[recall_index]

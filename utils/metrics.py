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
        if self.ensemble:
            sm = nn.LogSoftmax(dim=2)
            probs = sm(logits)
            probs = torch.mean(probs, dim=0)
        else:
            sm = nn.LogSoftmax(dim=1)
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
        # all_predictions = torch.cat(self.probabilities, dim=0)
        # all_labels = torch.cat(self.labels)
        #
        # one_hot_labels = F.one_hot(all_labels)
        #
        # fpr, tpr, _ = roc_curve(one_hot_labels.ravel(), all_predictions.ravel())
        #
        # recall_index = np.argmax(tpr >= 0.95)
        #
        # return fpr[recall_index]
        # fpr95 = FPR95()
        return 0

from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat

import numpy as np
from numpy.typing import ArrayLike

#
# def stable_cumsum(arr: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08):
#     """
#     From https://github.com/hendrycks/anomaly-seg
#     Uses high precision for cumsum and checks that the final value matches
#     the sum.
#     Args:
#     arr : array-like
#         To be cumulatively summed as flat
#     rtol : float
#         Relative tolerance, see ``np.allclose``
#     atol : float
#         Absolute tolerance, see ``np.allclose``
#     """
#     out = np.cumsum(arr, dtype=np.float64)
#     expected = np.sum(arr, dtype=np.float64)
#     if not np.allclose(
#             out[-1], expected, rtol=rtol, atol=atol
#     ):  # coverage: ignore
#         raise RuntimeError(
#             "cumsum was found to be unstable: "
#             "its last element does not correspond to sum"
#         )
#     return out
#
#
# class FPR95(Metric):
#     """Class which computes the False Positive Rate at 95% Recall."""
#
#     is_differentiable: bool = False
#     higher_is_better: bool = False
#     full_state_update: bool = False
#
#     conf: List[Tensor]
#     targets: List[Tensor]
#
#     def __init__(self, pos_label: int, **kwargs) -> None:
#         super().__init__(**kwargs)
#
#         self.pos_label = pos_label
#         self.add_state("conf", [], dist_reduce_fx="cat")
#         self.add_state("targets", [], dist_reduce_fx="cat")
#
#         rank_zero_warn(
#             "Metric `FPR95` will save all targets and predictions"
#             " in buffer. For large datasets this may lead to large memory"
#             " footprint."
#         )
#
#     def update(self, conf: Tensor, target: Tensor) -> None:  # type: ignore
#         self.conf.append(conf)
#         self.targets.append(target)
#
#
# def compute(self) -> Tensor:
#     r"""From https://github.com/hendrycks/anomaly-seg
#     Compute the actual False Positive Rate at 95% Recall.
#     Returns:
#         Tensor: The value of the FPR95.
#     """
#     conf = dim_zero_cat(self.conf).cpu().numpy()
#     targets = dim_zero_cat(self.targets).cpu().numpy()
#
#     # out_labels is an array of 0s and 1s - 0 if IOD 1 if OOD
#     out_labels = targets == self.pos_label
#
#     in_scores = conf[np.logical_not(out_labels)]
#     out_scores = conf[out_labels]
#
#     # pos = OOD
#     neg = np.array(in_scores[:]).reshape((-1, 1))
#     pos = np.array(out_scores[:]).reshape((-1, 1))
#     examples = np.squeeze(np.vstack((pos, neg)))
#     labels = np.zeros(len(examples), dtype=np.int32)
#     labels[: len(pos)] += 1
#
#     # make labels a boolean vector, True if OOD
#     labels = labels == self.pos_label
#
#     # sort scores and corresponding truth values
#     desc_score_indices = np.argsort(examples, kind="mergesort")[::-1]
#     examples = examples[desc_score_indices]
#     labels = labels[desc_score_indices]
#
#     # examples typically has many tied values. Here we extract
#     # the indices associated with the distinct values. We also
#     # concatenate a value for the end of the curve.
#     distinct_value_indices = np.where(np.diff(examples))[0]
#     threshold_idxs = np.r_[distinct_value_indices, labels.shape[0] - 1]
#
#     # accumulate the true positives with decreasing threshold
#     tps = stable_cumsum(labels)[threshold_idxs]
#     fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing
#
#     thresholds = examples[threshold_idxs]
#
#     recall = tps / tps[-1]
#
#     last_ind = tps.searchsorted(tps[-1])
#     sl = slice(last_ind, None, -1)  # [last_ind::-1]
#     recall, fps, tps, thresholds = (
#         np.r_[recall[sl], 1],
#         np.r_[fps[sl], 0],
#         np.r_[tps[sl], 0],
#         thresholds[sl],
#     )
#
#     cutoff = np.argmin(np.abs(recall - 0.95))
#
#     return torch.tensor(fps[cutoff] / (np.sum(np.logical_not(labels))), dtype=torch.float32)
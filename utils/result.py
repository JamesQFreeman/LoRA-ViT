import logging
import os
# import os
import time

import numpy as np
import torch
# import yaml
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score,recall_score,multilabel_confusion_matrix)


def mkdirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = torch.eye(num_cls, device=label.device)[label]
    return label


class ResultCLS:
    def __init__(self, num_cls) -> None:
        self.epoch = 1
        self.best_epoch = 0
        self.best_result = 0.0
        self.num_cls = num_cls
        return

    def eval(self, label, pred):
        self.pred.append(pred)
        self.true.append(label)
        return

    def init(self):
        self.st = time.time()
        self.pred = []
        self.true = []
        return

    @torch.no_grad()
    def stastic(self):
        num_cls = self.num_cls

        pred = torch.cat(self.pred, dim=0)
        true = torch.cat(self.true, dim=0)

        probe = torch.softmax(pred, dim=1).cpu().detach().numpy()
        true_one_hot = get_one_hot(true, num_cls).cpu().detach().numpy()
        true = true.cpu().detach().numpy()
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy()

        self.acc = accuracy_score(true, pred)
        self.sen = sensitivity_score(true, pred, average="macro")
        self.spe = specificity_score(true, pred, average="macro")
        self.pre = precision_score(true, pred, average="macro")
        self.f1 = f1_score(true, pred, average="macro")
        self.auc = roc_auc_score(true_one_hot, probe, average="macro")
        self.cm = confusion_matrix(true, pred)
        self.time = np.round(time.time() - self.st, 1)

        self.pars = [self.acc, self.sen, self.spe, self.pre, self.f1, self.auc]
        return

    

    def print(self, epoch: int, datatype='test'):
        self.stastic()
        titles = ["dataset", "ACC", "SEN", "SPE", "PRE", "F1S", "AUC"]
        items = [datatype.upper()] + self.pars
        forma_1 = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"
        logging.info(f"ACC: {self.pars[1]:.3f}, TIME: {self.time:.1f}s")
        logging.info((forma_1 + forma_2).format(*titles, *items))
        logging.debug(f"\n{self.cm}")
        self.epoch = epoch

        if self.acc > self.best_result:
            self.best_epoch = epoch
            self.best_result = self.acc
        return

    @torch.no_grad()
    def stastic_multi(self):
        num_cls = self.num_cls

        pred = torch.cat(self.pred, dim=0)
        true = torch.cat(self.true, dim=0)

        probe=torch.sigmoid(pred).cpu().detach().numpy()
        pred= (probe>0.5)
        true = true.cpu().detach().numpy()


        self.acc = accuracy_score(true, pred)
        # self.sen = sensitivity_score(true, pred, average="macro")
        # self.spe = specificity_score(true, pred, average="macro")
        self.pre = precision_score(true, pred, average="macro")
        self.rec = recall_score(true, pred, average="macro")
        self.f1 = f1_score(true, pred, average="macro")
        self.auc = roc_auc_score(true, probe, average="macro")
        self.cm = multilabel_confusion_matrix(true, pred)
        self.time = np.round(time.time() - self.st, 1)

        self.pars = [self.acc, self.pre, self.rec, self.f1, self.auc]
        return

    def print_multi(self, epoch: int, datatype='test'):
        self.stastic_multi()
        titles = ["dataset", "ACC", "PRE", "REC","F1", "AUC"]
        items = [datatype.upper()] + self.pars
        forma_1 = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"
        logging.info(f"ACC: {self.pars[1]:.3f}, TIME: {self.time:.1f}s")
        logging.info((forma_1 + forma_2).format(*titles, *items))
        logging.debug(f"\n{self.cm}")
        self.epoch = epoch
        if self.acc > self.best_result:
            self.best_epoch = epoch
            self.best_result = self.acc
        return

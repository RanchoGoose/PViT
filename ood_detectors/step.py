import torch
import torch
import torch.nn.functional as F
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score


class STEPOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        self.logits_train = train_model_outputs['logits']
        self.feas_train = train_model_outputs['feas']

    def infer(self, args, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']


        return torch.tensor(scores).to(logits[0].device)


import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score

class NNGuideOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):

        logits_train = train_model_outputs['logits']
        feas_train = train_model_outputs['feas']
        train_labels = train_model_outputs['labels']

        hyperparam = args.detector

        try:
            self.knn_k = hyperparam['knn_k']
        except:
            self.knn_k = 10

        confs_train = torch.logsumexp(logits_train, dim=1)
        print("feas_train shape:", feas_train.shape)
        print("confs_train shape:", confs_train.shape)
        self.scaled_feas_train = feas_train * confs_train[:, None]

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']
        print("scaled_feas_train shape:", self.scaled_feas_train.shape)

        confs = torch.logsumexp(logits, dim=1)
        guidances = knn_score(self.scaled_feas_train, feas, k=self.knn_k)
        scores = torch.from_numpy(guidances).to(confs.device)*confs
        return scores
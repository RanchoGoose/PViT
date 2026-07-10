import torch
import torch
import torch.nn.functional as F
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score

def jensen_shannon_divergence(p, q):
    # Calculate JS Divergence with epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    m = 0.5 * (p + q)

    js_divergence = 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') + F.kl_div(q.log(), m, reduction='batchmean'))
    return js_divergence


class PViTOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        self.logits_train = train_model_outputs['logits']
        self.feas_train = train_model_outputs['feas']
        self.priors_train = train_model_outputs['priors']
        # args.score = args.score

    def infer(self, args, model_outputs):
        """Compute per-sample detection scores.

        Score orientation convention: every score returned here is a
        CONFIDENCE score -- the higher the score, the more ID-like the
        input. This matches the evaluation harness: main.py labels ID
        samples as the positive class (1) and
        eval_assets.compute_ood_performances consumes the scores
        unmodified.

        The paper's PGE score S_PGE = S_base * G (Eq. 12) is an OOD-ness
        score (small for ID, large for OOD, since the guidance term G is
        a divergence). Accordingly, the PGE-based options below return
        -S_PGE, which is equivalent to the decision rule
        "x is ID iff S_PGE < gamma" (Eq. 13).
        """
        feas = model_outputs['feas']
        logits = model_outputs['logits']
        priors = model_outputs['priors']

        # Initialize scores list
        scores = []

        if args.score == 'knn':
            scores = knn_score(self.feas_train, feas, k=10, min=True)

        elif args.score == 'nnguide':
            confs_train = torch.logsumexp(self.logits_train, dim=1)
            scaled_feas_train = self.feas_train * confs_train[:, None]
            confs = torch.logsumexp(logits, dim=1)
            guidances = knn_score(scaled_feas_train, feas, k=10)
            scores = torch.from_numpy(guidances).to(confs.device)*confs

        elif args.score == 'energy':
            scores = torch.logsumexp(logits, dim=1)

        else:
            # Iterate over each element in the list
            for i in range(len(logits)):
                # Convert each element to a tensor if they aren't already
                current_logits = torch.tensor(logits[i]) if isinstance(logits[i], list) else logits[i]
                current_priors = torch.tensor(priors[i]) if isinstance(priors[i], list) else priors[i]
                current_feas = torch.tensor(feas[i]) if isinstance(feas[i], list) else feas[i]

                # Ensure dimensions are correct
                current_logits = current_logits.unsqueeze(0) if current_logits.dim() == 1 else current_logits
                current_priors = current_priors.unsqueeze(0) if current_priors.dim() == 1 else current_priors

                _, pred = torch.max(current_logits, dim=-1)
                # Base confidence score S_base = -E(x; theta) (Eq. 8), computed
                # from PViT's own logits; positive for practical logit ranges.
                s_base = torch.logsumexp(current_logits, dim=-1)

                if args.score == 'double_energy':
                    prior_energy = torch.logsumexp(current_priors, dim=1)
                    true_energy = torch.logsumexp(current_logits, dim=1)
                    score = prior_energy + true_energy

                elif args.score == 'cosine':
                    # Cosine similarity between prior and predicted logits:
                    # high similarity = agreement = ID-like.
                    score = F.cosine_similarity(current_priors, current_logits, dim=1)

                elif args.score == 'cross_entropy':
                    if pred.shape[0] != current_priors.shape[0]:
                        pred = pred.expand(current_priors.shape[0])

                    # Guidance term G (Eq. 11): cross-entropy from PViT's
                    # predicted class to the prior distribution, -log q_c.
                    # Raw prior logits are passed because F.cross_entropy
                    # applies log_softmax internally.
                    guidance = F.cross_entropy(current_priors, pred, reduction='none')
                    score = -s_base * guidance  # confidence = -S_PGE

                elif args.score == 'dis':
                    # Guidance term G_ED: Euclidean distance between prior
                    # and predicted logits.
                    guidance = torch.norm(current_priors - current_logits, dim=1)
                    score = -s_base * guidance  # confidence = -S_PGE

                elif args.score == 'KL':
                    # Guidance term G_KL = KL(P_prior || Q_pvit).
                    q = F.softmax(current_logits, dim=1)
                    p = F.softmax(current_priors, dim=1)
                    guidance = F.kl_div(torch.log(q), p, reduction='none').sum(dim=1)
                    score = -s_base * guidance  # confidence = -S_PGE

                elif args.score == 'guidance_only':
                    # Ablation: the CE guidance term alone, without S_base.
                    if pred.shape[0] != current_priors.shape[0]:
                        pred = pred.expand(current_priors.shape[0])
                    score = -F.cross_entropy(current_priors, pred, reduction='none')

                elif args.score == 'additive':
                    # Ablation: additive combination S_base - G.
                    if pred.shape[0] != current_priors.shape[0]:
                        pred = pred.expand(current_priors.shape[0])
                    guidance = F.cross_entropy(current_priors, pred, reduction='none')
                    score = s_base - guidance

                elif args.score == 'difference':
                    p = F.softmax(current_priors, dim=1)
                    score = -(p.argmax(dim=1) != current_logits.argmax(dim=1)).float()

                elif args.score == 'JS':
                    current_priors = F.softmax(current_priors, dim=1)
                    current_logits = F.softmax(current_logits, dim=1)
                    score = -jensen_shannon_divergence(current_priors, current_logits)

                scores.append(score.item())

        return torch.tensor(scores).to(logits[0].device)


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
                energy = torch.logsumexp(current_priors, dim=-1)
                    
                if args.score == 'double_energy':
                    prior_energy = torch.logsumexp(current_priors, dim=1)
                    true_energy = torch.logsumexp(current_logits, dim=1)
                    score = prior_energy + true_energy

                elif args.score == 'cosine':
                    score = -F.cosine_similarity(current_priors, current_logits, dim=1)
                    
                    
                elif args.score == 'cross_entropy':
                    if pred.shape[0] != current_logits.shape[0]:
                        pred = pred.expand(current_logits.shape[0])
                        
                    priorsout = torch.softmax(current_priors, dim=1)
                    # energy = torch.logsumexp(current_priors, dim=-1)
                    score = F.cross_entropy(priorsout, pred, reduction='none')
                    # score = score * energy
                    
                # elif args.score == 'cross_entropy':              
                #     # Ensure `pred` matches the batch size
                #     if pred.shape[0] != current_logits.shape[0]:
                #         pred = pred.expand(current_logits.shape[0])

                #     score = -F.cross_entropy(current_logits, pred, reduction='none')

                #     # # Optional: Multiply scores by prior energy
                #     # prior_energy = torch.logsumexp(current_priors, dim=-1)
                #     # score = score * prior_energy
                    
                elif args.score == 'dis':
                    # energy = torch.logsumexp(current_priors, dim=-1)
                    score = torch.norm(current_priors - current_logits, dim=1)
                    # score = score * energy

                elif args.score == 'KL':
                    q = F.softmax(current_logits, dim=1)
                    p = F.softmax(current_priors, dim=1)
                    score = -F.kl_div(torch.log(q), p, reduction='none').sum(dim=1)
                    # score = score * energy

                elif args.score == 'difference':
                    p = F.softmax(current_priors, dim=1)
                    score = -(p.argmax(dim=1) != current_logits.argmax(dim=1)).float()
                    
                elif args.score == 'JS':
                    current_priors = F.softmax(current_priors, dim=1)
                    current_logits = F.softmax(current_logits, dim=1)
                    score = -jensen_shannon_divergence(current_priors, current_logits)
                    # score = score * energy
                    
                # score = score * energy
                # Append score to the list
                scores.append(score.item())  # or just append(score) for tensor output

        return torch.tensor(scores).to(logits[0].device)


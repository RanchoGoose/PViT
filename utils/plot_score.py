import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('../results/True_SVHN_CIFAR10_1.0_small_vit_cross_entropy_10_scores.csv')

# Sample 100 data points from each score column
sampled_in_score = data['In-Score'].sample(100, random_state=42)
sampled_out_score = data['Out-Score'].sample(100, random_state=42)

# # Calculate metrics
# conf = np.concatenate((sampled_in_score, sampled_out_score))
# ind_indicator = np.concatenate((np.ones_like(sampled_in_score), np.zeros_like(sampled_out_score)))
# fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
# auroc = metrics.auc(fpr, tpr)
#
# # Calculate FPR95
# fpr95, _ = fpr_recall(sampled_in_score, sampled_out_score, 0.95)

# Plot the Kernel Density Estimates for the sampled scores
plt.figure(figsize=(12, 6))
sns.kdeplot(sampled_in_score, label='In-Score', color='blue', fill=True)
sns.kdeplot(sampled_out_score, label='Out-Score', color='grey', fill=True)
# plt.title('OOD Scores')
plt.xlabel('PCE Score', fontsize=18)  # Adjust as needed
plt.ylabel('Density', fontsize=18)  # Adjust as needed
plt.legend(loc='upper right', fontsize=18)  # Adjust as needed

# Add AUROC and FPR95 as title
plt.suptitle(f'ID: CIFAR10, OOD: SVHN', fontsize=18)

plt.show()





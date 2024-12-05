import numpy as np
from sklearn import metrics

from misc.utils_python import dict2csv

def compute_ood_performances(labels, scores):
    # labels: 0 = OOD, 1 = ID
    # scores: it is anomality score (the higher the score, the more anomalous)

    # auroc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)

    # tnr at tpr 95
    idx_tpr95 = np.abs(tpr - .95).argmin()
    fpr_at_tpr95 = fpr[idx_tpr95]

    # dtacc (detection accuracy)
    dtacc = .5 * (tpr + 1. - fpr).max()

    # auprc (in and out)
    # ref. https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/utils.py#L91
    auprc_in = metrics.average_precision_score(y_true=labels, y_score=scores)
    auprc_out = metrics.average_precision_score(y_true=labels, y_score=-scores, pos_label=0) # equivalent to average_precision_score(y_true=1-labels, y_score=-scores)

    return auroc, fpr_at_tpr95, dtacc, auprc_in, auprc_out


import pandas as pd
def save_performance(scores_set, labels, accs, log_path):
    pfs = {}
    pfs['auroc'] = {}
    pfs['fpr'] = {}
    pfs['dtacc'] = {}
    pfs['auin'] = {}
    pfs['auout'] = {}

    for k in scores_set.keys():
        pfs['auroc'][k], pfs['fpr'][k], pfs['dtacc'][k], pfs['auin'][k], pfs['auout'][k] \
            = compute_ood_performances(labels, scores_set[k])

    for k_m in pfs.keys():
        for k_s in pfs[k_m].keys():
            pfs[k_m][k_s] *= 100

    pfs['acc'] = {}
    for key, acc in accs.items():
        pfs['acc'][key] = acc

    '''
    save result
    '''
    df = pd.DataFrame(pfs)
    df.round(2)
    print(df)
    df.to_csv(log_path)
    # for key in pfs.keys():
    #     dict2csv(pfs[key], f"{log_dir_path}/{key}_split-{split_idx}.csv")
    
    
    
def save_raw_scores(scores_set, labels, raw_scores_path):
    """
    Save raw scores along with labels to a CSV file for later analysis or plotting.

    Parameters:
    - scores_set: dict
        A dictionary where keys are model names and values are lists of scores.
    - labels: list
        A list of ground truth labels indicating ID (0) or OOD (1).
    - raw_scores_path: str
        The file path to save the raw scores CSV.
    """
    # Create a DataFrame from scores_set
    df_scores = pd.DataFrame.from_dict(scores_set, orient='index').transpose()
    
    # Add labels to the DataFrame
    df_scores['labels'] = labels
    
    # Save the raw scores DataFrame to a CSV file
    df_scores.to_csv(raw_scores_path, index=False)
    print("Raw scores with labels saved to:", raw_scores_path)
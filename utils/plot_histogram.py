import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# Example data loading
# Replace this with your actual data loading process

# Assuming you have columns like 'score_type_1', 'score_type_2', ..., 'labels'
score_types = ['Energy', 'KL', 'Euclidean distance', 'PCE']
# auroc_values = [93.15, 92.67, 81.03, 95.84]
# fpr_values = [24.56, 22.54, 37.66, 22.21]

auroc_values = [93.94, 71.32, 81.03, 94.32]
fpr_values = [23.57, 64.24, 37.66, 22.86]

x = np.arange(len(score_types))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting AUROC values
bars1 = ax1.bar(x - width/2, auroc_values, width, label='AUROC', color='grey')

# Adding values above AUROC bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}',
             verticalalignment='bottom', ha='center', fontsize=12)

ax1.set_xlabel('Score Types', fontsize=20)
ax1.set_ylabel('AUROC', fontsize=20, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(score_types, fontsize=18)

# Instantiate a second y-axis for FPR values
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, fpr_values, width, label='FPR', color='orange')

# Adding values above FPR bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}',
             verticalalignment='bottom', ha='center', fontsize=12)

ax2.set_ylabel('FPR', fontsize=20, color='black')
ax2.tick_params(axis='y', labelcolor='black')

fig.tight_layout()
plt.savefig('score1.png')
plt.show()

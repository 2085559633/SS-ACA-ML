import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ------------------------------------------------------------------
# Files & basic settings
# ------------------------------------------------------------------
train_csv = 'all_train_results.csv'
val_csv   = 'all_val_results.csv'
label_cols = {'train': 'y_train', 'val': 'y_val'}

output_dir = Path('rocplot')          # 当前目录下的子文件夹
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 计算 AUC 及 95% CI（bootstrap）
# ------------------------------------------------------------------
def bootstrap_auc(y_true, y_score, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    boot_scores = []

    for _ in range(n_bootstraps):
        # 有放回抽样生成同样大小的索引
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            # 这一轮抽样没有正或负样本，跳过
            continue
        score = roc_auc_score(y_true[indices], y_score[indices])
        boot_scores.append(score)

    boot_scores = np.array(boot_scores)
    boot_scores.sort()

    mean_auc = roc_auc_score(y_true, y_score)
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    return mean_auc, lower, upper


# ------------------------------------------------------------------
# 绘制并保存单张 ROC
# ------------------------------------------------------------------
def save_single_roc(df, label_col, proba_col, set_name):
    y_true  = df[label_col].values
    y_score = df[proba_col].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    mean_auc, ci_low, ci_up = bootstrap_auc(y_true, y_score)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2,
             label=f'AUC = {mean_auc:.3f} (95% CI {ci_low:.3f}–{ci_up:.3f})')
    plt.plot([0, 1], [0, 1], '--', lw=1)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC – {proba_col} ({set_name})')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    pdf_name = f'{proba_col}_{set_name}_ROC.pdf'.replace('__', '_')
    out_path = output_dir / pdf_name
    plt.savefig(out_path, format='pdf', dpi=300)
    plt.close()
    return out_path


# ------------------------------------------------------------------
# 主流程：读数据 & 批量绘制
# ------------------------------------------------------------------
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
proba_cols = [c for c in train_df.columns if c.endswith('_proba')]

generated = []
for col in proba_cols:
    generated.append(save_single_roc(train_df, label_cols['train'], col, 'train'))
    generated.append(save_single_roc(val_df,   label_cols['val'],   col, 'val'))

print(f'Generated {len(generated)} PDF files in {output_dir.resolve()}')
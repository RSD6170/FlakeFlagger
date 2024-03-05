import os
from pathlib import Path

import pandas as pd
import numpy as np


def extract_parameters(path: str):
    return path.split("/")


root_path = "/home/ubuntu/atsfp/atsfp-23-24/data/fst_with_multiclass/classification_result"

file_dict = {}

for path, dirs, files in os.walk(root_path):
    if not files: continue
    file_dict[str(Path(path).relative_to(root_path))] = {Path(file).stem : pd.read_csv(os.path.join(path, file), index_col=0) for file in files}

data = []

for k,v in file_dict.items():
    accuracy = v["accuracy_norm"].iloc[-1,-1]
    micro_auc = v["full_report"].loc["avg / total", "AUC"]
    micro_f1 = v["full_report"].loc["avg / total", "f1-score"]
    micro_prec = v["full_report"].loc["avg / total", "precision"]
    micro_rec = v["full_report"].loc["avg / total", "recall"]
    roc_auc_ovo_weighted = v["roc_auc_ovo_weighted"].iloc[-1,-1]
    data.append(extract_parameters(k) + [accuracy, micro_auc, micro_f1, micro_prec, micro_rec, roc_auc_ovo_weighted])



result = pd.DataFrame(data, columns=["IG", "Fold_Type", "Balancer", "Imputer_Strategy", "Classifier", "Tree_Depth", "Strategy", "accuracy", "micro_auc", "micro_f1", "micro_prec", "micro_rec", "roc_auc_ovo_weighted"])
result.to_csv(root_path + "/comparison.csv", index=False)
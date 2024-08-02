import seaborn as sn
import pandas as pd
from pretty_confusion_matrix import pp_matrix
import matplotlib.pyplot as plt



df_cm = pd.read_csv("/home/ubuntu/atsfp/atsfp-23-24/data/fst_with_multiclass/classification_result/0.01/StratifiedKFold/both/most_frequent/RF/1000/both/confusion.csv", index_col=0)
# plt.figure(figsize=(10,7))
cmap = "Oranges_r"
pp_matrix(df_cm, show_null_values=1, cmap=cmap)

#plt.savefig("confusion.pdf")

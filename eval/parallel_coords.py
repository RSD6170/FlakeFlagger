import pandas as pd
import plotly.express as px


df = pd.read_csv("/home/ubuntu/atsfp/atsfp-23-24/data/fst_with_multiclass/classification_result/comparison.csv", index_col=0).reset_index()
df = df.drop(['accuracy','micro_auc','micro_f1','micro_prec','micro_rec'], axis=1)

bin = 100
bins = [i/bin for i in range(bin + 1)]

df['color'] = pd.cut(df['roc_auc_ovo_weighted'], bins, labels=False)



fig = px.parallel_categories(df,
    dimensions=['IG','Fold_Type','Balancer','Imputer_Strategy','Classifier','Tree_Depth','Strategy'],
                             color="color",
                             color_continuous_scale=px.colors.diverging.Tealrose)
fig.write_html("test.html")
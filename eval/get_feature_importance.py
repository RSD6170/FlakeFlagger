import io
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import warnings
import numpy as np
import os
import sys
from pathlib import Path
import ast

import sklearn.model_selection
from imblearn.pipeline import Pipeline
from matplotlib import cm
from numpy import interp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, \
    make_scorer, precision_recall_fscore_support, accuracy_score, RocCurveDisplay
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import svm, tree
import math

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import logging

import matplotlib.pyplot as plt

# from https://stackoverflow.com/a/50417060
def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
            y_true.shape,
            y_pred.shape)
              )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


def single_combiner(a, b):
    if np.isnan(a):
        return b
    elif np.isnan(b):
        return a
    else:
        return a + b


def series_combiner(a_series, b_series):
    return a_series.combine(b_series, single_combiner, fill_value=np.nan)


def dataframe_combiner(a_df, b_df):
    return a_df.combine(b_df, series_combiner, fill_value=np.nan)


# %%
def predict_RF_crossValidation(data, k, foldType, imputer_strategy_loc, balance, classifier, mintree, name):
    data_target = data['flaky']
    data = data.drop(['flaky', 'test_name', 'project_y', 'project'], axis=1, errors='ignore')

    # KFold Cross Validation approaches
    if (foldType == "KFold"):
        fold = KFold(n_splits=k)
    else:
        fold = StratifiedKFold(n_splits=k)

    pipeline_list = []

    pipeline_list.append(('imputer', SimpleImputer(strategy=imputer_strategy_loc)))

    if (balance == "SMOTE"):
        oversample = SMOTE()
        pipeline_list.append(('pre_SMOTE', oversample))
    elif (balance == "undersampling"):
        undersampling = RandomUnderSampler()
        pipeline_list.append(('pre_under', undersampling))
    elif (balance == "both"):
        oversample = SMOTE()
        pipeline_list.append(('pre_SMOTE', oversample))
        undersampling = RandomUnderSampler()
        pipeline_list.append(('pre_under', undersampling))

    if (classifier == 'DT'):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=None)
        pipeline_list.append(('model_DT', model))
    elif (classifier == 'RF'):
        model = RandomForestClassifier(criterion="entropy", n_estimators=mintree, n_jobs=12)
        pipeline_list.append(('model_RF', model))
    elif (classifier == 'MLP'):
        model = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=50)
        pipeline_list.append(('model_MLP', model))
    elif (classifier == 'SVM'):
        model = svm.SVC(gamma='scale')
        pipeline_list.append(('model_SVM', model))
    elif (classifier == 'Ada'):
        model = AdaBoostClassifier(n_estimators=100, random_state=0)
        pipeline_list.append(('model_Ada', model))
    elif (classifier == 'NB'):
        model = GaussianNB()
        pipeline_list.append(('model_NB', model))
    elif (classifier == 'KNN'):
        model = KNeighborsClassifier(n_neighbors=7)
        pipeline_list.append(('model_KNN', model))

    pipeline = Pipeline(pipeline_list)

    labels = data_target.unique().tolist()
    n_classes = len(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, data_target, test_size=0.3, random_state=0)
    y_score = pipeline.fit(X_train, y_train)

    std = np.std([tree.feature_importances_ for tree in pipeline.named_steps['model_RF'].estimators_], axis=0)
    importances = pipeline.named_steps['model_RF'].feature_importances_
    forest_importances = pd.DataFrame({'importance':importances, 'std':std, 'label':list(X_test.columns)})
    forest_importances.sort_values('importance', inplace=True, ascending=False)


    fig, ax = plt.subplots(figsize=(9, 7))
    forest_importances.head(10).plot.barh(x='label', y='importance', xerr='std', ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_xlabel("")
    fig.tight_layout()

    plt.savefig(f"importance_{name}.pdf")


# %%
def get_only_specific_columns_V1(copy_fullData, specificColumns, wanted_columns):
    lst = []
    for i in specificColumns:
        lst.append(i)
    for j in wanted_columns:
        lst.append(j)
    available_columns = list(set(lst) & set(copy_fullData.columns))
    copy_fullData = copy_fullData[available_columns]
    return copy_fullData


# %%
def get_only_specific_columns_V2(copy_fullData, removed_specificColumns, removed_columns):
    lst = []
    for i in removed_specificColumns:
        lst.append(i)
    for j in removed_columns:
        lst.append(j)
    available_columns = list(set(lst) & set(copy_fullData.columns))
    copy_fullData = copy_fullData.drop(columns=available_columns)
    return copy_fullData


# %%
def get_only_specific_columns_V3(copy_fullData, removed_columns):
    lst = []
    for j in removed_columns:
        lst.append(j)
    available_columns = list(set(lst) & set(copy_fullData.columns))
    copy_fullData = copy_fullData.drop(columns=available_columns)
    return copy_fullData


# %%
def vexctorizeToken(token):
    vocabulary_vectorizer = CountVectorizer()
    bow_train = vocabulary_vectorizer.fit_transform(token)
    matrix_token = pd.DataFrame(bow_train.toarray(), columns=vocabulary_vectorizer.get_feature_names_out())

    return matrix_token


def export_dict(path, ret_dict):
    Path(path).mkdir(parents=True, exist_ok=True)
    for k, v in ret_dict.items():
        v.to_csv(path + f"{k}.csv")


def analyse_config(k, ig, fold, bal, imp_strategy, cl, mintree, vocabulary_processed_data_full, FlakeFlaggerFeatures,
                   output_dir, removed_columns):
    # print the given variables for easy debug.
    print(
        f"==> run selection is: (information_gain>={ig})+(Classifier={cl})+(Balance={bal})+(Imputer Strategy={imp_strategy})+(Fold type={fold})+(Minimum tress [RF only]={mintree}")

    subpath = f"{ig}/{fold}/{bal}/{imp_strategy}/{cl}/{mintree}/"

    eval_flakeflagger(FlakeFlaggerFeatures, bal, cl, fold, imp_strategy, k, mintree, output_dir, subpath, vocabulary_processed_data_full)

    eval_dict(FlakeFlaggerFeatures, bal, cl, fold, imp_strategy, k, mintree, output_dir, removed_columns, subpath, vocabulary_processed_data_full)

    eval_both(bal, cl, fold, imp_strategy, k, mintree, output_dir, removed_columns, subpath, vocabulary_processed_data_full)

    print("=======================================================================")


def eval_both(bal, cl, fold, imp_strategy, k, mintree, output_dir, removed_columns, subpath,
              vocabulary_processed_data_full):
    # get only vocabulary features ..
    eval = predict_RF_crossValidation(
        get_only_specific_columns_V3(vocabulary_processed_data_full, removed_columns),
        k, fold, imp_strategy, bal, cl, mintree,"both")

    print("--> The prediction based on the FlakeFlagger with vocabulary features is completed ")


def eval_dict(FlakeFlaggerFeatures, bal, cl, fold, imp_strategy, k, mintree, output_dir, removed_columns, subpath,
              vocabulary_processed_data_full):
    # get only vocabulary features ..
    eval = predict_RF_crossValidation(
        get_only_specific_columns_V2(vocabulary_processed_data_full,
                                     FlakeFlaggerFeatures.allFeatures.unique(), removed_columns),
        k, fold, imp_strategy, bal, cl, mintree,"dict")

    print("--> The prediction based on the collected vocabulary only is completed ")


def eval_flakeflagger(FlakeFlaggerFeatures, bal, cl, fold, imp_strategy, k, mintree, output_dir, subpath,
                      vocabulary_processed_data_full):
    # get only FlakeFlagger features ..
    eval = predict_RF_crossValidation(
        get_only_specific_columns_V1(vocabulary_processed_data_full,
                                     FlakeFlaggerFeatures.allFeatures.unique(), ["flaky", "test_name"]),
        k, fold, imp_strategy, bal, cl, mintree,"ff")

    print("--> The prediction based on the FlakeFlagger features is completed ")


def generate_vocab_processed_data(path):
    main_data = pd.read_csv(path)
    tokenOnly = vexctorizeToken(main_data['tokenList'])
    main_data = main_data.drop(columns=['tokenList'])
    return pd.concat([main_data, tokenOnly.reindex(main_data.index)], axis=1)


# %%
execution_time = time.time()
# command : python3 cross-all-projects-model-vocabulary.py input_data/data/full_data.csv input_data/FlakeFlaggerFeaturesTypes.csv token_by_IG/IG_vocabulary_and_FlakeFlagger_features.csv

if __name__ == '__main__':
    # pd.set_option("mode.copy_on_write", True)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    warnings.simplefilter("ignore")

    root = "/home/ubuntu/atsfp/atsfp-23-24/data/fst_with_multiclass/"

    # name of FlakeFlaggerFeatures ..
    FlakeFlaggerFeatures = pd.read_csv(
        "/home/ubuntu/atsfp/FlakeFlagger/flakiness-predicter/input_data/FlakeFlaggerFeaturesTypes.csv")

    # IG per token/FlakeFlagger/JavaKeyWords
    IG_lst = pd.read_csv(root + "Information_gain_per_feature.csv")

    output_dir = root + "/classification_result/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # vocabulary data _ processed data
    vocabulary_processed_data = generate_vocab_processed_data(root + "processed_data_with_vocabulary_per_test.csv")


    ##=========================================================##
    # arguments
    k = 10  # number of folds
    fold_type = ["StratifiedKFold"]
    balance = ["both"]
    classifier = ["RF"]
    treeSize = [1000]
    minIGList = [0.01]
    imputer_strategy = ['most_frequent']
    ##=========================================================##

    with (ProcessPoolExecutor(max_workers=10) as executor):
        for ig in minIGList:
            min_IG = IG_lst[IG_lst["IG"] >= ig]
            keep_minIG = min_IG.features.unique()
            keep_minIG = [x for x in keep_minIG if str(x) != 'nan']
            removed_columns = ['java_keywords', 'javaKeysCounter', 'Java_keywords']

            vocabulary_processed_data_full = vocabulary_processed_data
            if (ig != 0):
                keep_columns = keep_minIG + ['flaky', 'test_name']
                vocabulary_processed_data_full = vocabulary_processed_data_full[keep_columns]

            futures = {}
            for fold in fold_type:
                for bal in balance:
                    for imp_strategy in imputer_strategy:
                        for cl in classifier:
                            loc_tree = treeSize
                            if cl != "RF":
                                loc_tree = [0]
                            for mintree in loc_tree:
                                analyse_config(k, ig, fold, bal, imp_strategy, cl, mintree, vocabulary_processed_data_full, FlakeFlaggerFeatures, output_dir, removed_columns)


print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))



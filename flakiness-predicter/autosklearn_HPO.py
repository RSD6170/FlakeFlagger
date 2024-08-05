import logging
import warnings
from pathlib import Path
from typing import Optional, Union, Any

import autosklearn.classification
import pandas as pd
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.feature_preprocessing import add_preprocessor
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, SIGNED_DATA

# paths
root = "/home/ubuntu/atsfp/atsfp-23-24/data/fst_with_multiclass/"
path_featureTypes = "/home/ubuntu/atsfp/FlakeFlagger/flakiness-predicter/input_data/FlakeFlaggerFeaturesTypes.csv"
path_IGlist = root + "Information_gain_per_feature.csv"
output_dir = root + "/classification_result/"
path_processedData = root + "processed_data_with_vocabulary_per_test.csv"


class IGFilter(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, IGbarrier):
        super().__init__()
        self.keep_minIG = None
        self.IGbarrier = IGbarrier
        self.igList = pd.read_csv(path_IGlist)

    @staticmethod
    def get_hyperparameter_search_space(feat_type: Optional[dict[Union[str, int], str]] = None,
                                        dataset_properties: Any = None):
        cs = ConfigurationSpace()
        barrier = UniformFloatHyperparameter(name="IGbarrier", lower=0.0, upper=1.0, default_value=0.01)
        cs.add_hyperparameter(barrier)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "IGBarrier",
            "name": "Information Gain Barrier per Feature",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    def fit(self, X, y):
        self.IGbarrier = float(self.IGbarrier)
        min_IG = IG_lst[IG_lst["IG"] >= self.IGbarrier]
        self.keep_minIG = min_IG.features.unique()
        self.keep_minIG = [x for x in self.keep_minIG if str(x) != 'nan'] + ['flaky', 'test_name']
        return self

    def transform(self, X):
        return X[self.keep_minIG]


class MethodFilter(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, type):
        super().__init__()
        self.type = type
        self.toDrop = []
        self.flakeFlaggerFeatures = pd.read_csv(path_featureTypes)

    @staticmethod
    def get_hyperparameter_search_space(feat_type: Optional[dict[Union[str, int], str]] = None,
                                        dataset_properties: Any = None):
        cs = ConfigurationSpace()
        type = CategoricalHyperparameter(name="type", choices=['ff', 'dict', 'both'], default_value='both')
        cs.add_hyperparameter(type)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "FeatureSelection",
            "name": "Choose Feature Strategy",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    def fit(self, X, y):
        removed_columns = ['java_keywords', 'javaKeysCounter', 'Java_keywords']
        if self.type == "ff":
            self.toDrop = ["flaky", "test_name"]
            self.toDrop.extend(self.flakeFlaggerFeatures.allFeatures.unique())
        elif self.type == "dict":
            self.toDrop = removed_columns
            self.toDrop.extend(self.flakeFlaggerFeatures.allFeatures.unique())
        elif self.type == "both":
            self.toDrop = removed_columns
        else:
            raise ValueError("Unexpected Type")
        return self

    def transform(self, X):
        if self.type == "ff":
            removeColumns = list(set(self.toDrop) & set(X.columns))
            return X[removeColumns]
        else:
            available_columns = list(set(self.toDrop) & set(X.columns))
            return X.drop(columns=available_columns)


if __name__ == '__main__':
    # pd.set_option("mode.copy_on_write", True)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    warnings.simplefilter("ignore")

    # IG per token/FlakeFlagger/JavaKeyWords
    IG_lst = pd.read_csv(path_IGlist)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # vocabulary data _ processed data
    vocabulary_processed_data = pd.read_csv(path_processedData)

    add_preprocessor(IGFilter)
    add_preprocessor(MethodFilter)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=600,
        resampling_strategy="cv",
        resampling_strategy_arguments={
                "train_size": 0.8,     # The size of the training set
                "shuffle": True,        # Whether to shuffle before splitting data
                "folds": 5              # Used in 'cv' based resampling strategies
            },
        n_jobs=1,
        metric=autosklearn.metrics.f1_weighted,
    )
    data_target = vocabulary_processed_data[['flaky']]
    data = vocabulary_processed_data.drop(['flaky', 'test_name', 'project_y', 'project'], axis=1, errors='ignore')
    automl.fit(data, data_target, dataset_name="IDoFT_Multi")
    print(automl.leaderboard())
    print(automl.show_models(), indent=4)


from typing import List
import pyarrow.feather as feather
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from config.config import Config
from utils import *


class CorrelationGenerator():
    def __init__(self) -> None:  
        self.features: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'undersample_OneSidedSelection_X.feather')
        self.target: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'undersample_OneSidedSelection_y.feather')
        # self.X_train, self.X_hold, self.X_eval, self.X_test, self.X_train_oh, self.X_eval_oh, self.X_test_oh, self.y_train, \
        #    self.y_hold,  self.y_eval, self.y_test = split_dataset(self.features, self.target)
        self.cluster_threshold = 1

    def serialize_dataset(self, df, file_name) -> None:
        print("Serializing at: ", Config.DATASETS_FOLDER + "\\correlation\\" + file_name)
        feather.write_feather(df, Config.DATASETS_FOLDER + "\\correlation\\" + file_name)

    def correlation_pipeline(self):
        corr = spearmanr(self.features).correlation
        corr_linkage = hierarchy.ward(corr)

        cluster_ids = hierarchy.fcluster(corr_linkage, self.cluster_threshold, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)

        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

        selected_features = self.features.columns[selected_features].tolist()
        print(self.features[selected_features])
        self.serialize_dataset(self.features[selected_features], "correlation.feather")


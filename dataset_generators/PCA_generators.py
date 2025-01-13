from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pyarrow.feather as feather
import time

from config.config import Config
from utils import *

class pcaGenerator():
    def __init__(self) -> None:
        # self.pca_component: List = [1, 0.95, 0.9, 0.85, 0.8]
        self.pca_component: List = [0.8]    
        self.features: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'features_dataset.feather')
        print(self.features)
        self.target: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'target_dataset.feather')
        self.X_train, self.X_hold, self.X_eval, self.X_test, self.X_train_oh, self.X_eval_oh, self.X_test_oh, self.y_train, \
            self.y_hold,  self.y_eval, self.y_test = split_dataset(self.features, self.target)

    def scale(self, X):
        scalar = StandardScaler()
        df_scaled = pd.DataFrame(scalar.fit_transform(X), columns=X.columns)
        return df_scaled
    
    def serialize_dataset(self, df, file_name, component_folder) -> None:
        print("Serializing at: ", Config.DATASETS_FOLDER + "\\" + component_folder + "\\" + file_name)
        feather.write_feather(df, Config.DATASETS_FOLDER + "\\" + component_folder + "\\" + file_name)

    def pipeline_binary(self):
        self.X_train, self.X_hold, self.X_eval, self.X_test, self.X_train_oh, self.X_eval_oh, self.X_test_oh
        X_train_scaled = self.scale(self.X_train)
        X_hold_scaled = self.scale(self.X_hold)
        X_eval_scaled = self.scale(self.X_eval)
        X_test_scaled = self.scale(self.X_test)
        X_train_oh_scaled = self.scale(self.X_train_oh)
        X_eval_oh_scaled = self.scale(self.X_eval_oh)
        X_test_oh_scaled = self.scale(self.X_test_oh)
        
        for item in self.pca_component:
            start_time = time.time()
            print(item)
            print(type(item))
            pca = PCA(item)
            X_train_pca = pd.DataFrame(pca.fit_transform(X_train_scaled))
            pca = PCA(item)
            X_hold_pca = pd.DataFrame(pca.fit_transform(X_hold_scaled))
            pca = PCA(item)
            X_eval_pca = pd.DataFrame(pca.fit_transform(X_eval_scaled))
            pca = PCA(item)
            X_test_pca = pd.DataFrame(pca.fit_transform(X_test_scaled))
            pca = PCA(item)
            X_train_oh_pca = pd.DataFrame(pca.fit_transform(X_train_oh_scaled))
            pca = PCA(item)
            X_eval_oh_pca = pd.DataFrame(pca.fit_transform(X_eval_oh_scaled))
            pca = PCA(item)
            X_test_oh_pca = pd.DataFrame(pca.fit_transform(X_test_oh_scaled))
            print("---", item, "---")
            print("--- %s seconds---" % (time.time() - start_time))

            # serialize
            self.serialize_dataset(X_train_pca, 'X_train_pca.feather', str(item))
            self.serialize_dataset(X_hold_pca, 'X_hold_pca.feather', str(item))
            self.serialize_dataset(X_eval_pca, 'X_eval_pca.feather', str(item))
            self.serialize_dataset(X_test_pca, 'X_test_pca.feather', str(item))
            self.serialize_dataset(X_train_oh_pca, 'X_train_oh_pca.feather', str(item))
            self.serialize_dataset(X_eval_oh_pca, 'X_eval_oh_pca.feather', str(item))
            self.serialize_dataset(X_test_oh_pca, 'X_test_oh_pca.feather', str(item))
            




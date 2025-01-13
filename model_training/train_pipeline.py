import pyarrow.feather as feather

from model_training.baseline_classifier import BaselineClassifier
from model_training.logistic_regression import LogisticRegressionClassifier
from model_training.random_forest import RFClassifier
from model_training.catboost_classifier import CatClassifier
from model_training.dl import DeepLearningClassifier

from config.config import Config
from utils import *

class TrainPipeline():
    def __init__(self) -> None:
        self.target: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\undersample_OneSidedSelection_y.feather")
        self.target['label_is_attack'] = self.target['label_cat'].apply(lambda x: 0 if x == 0 else 1)
        print(self.target)
        self.features_correlated = pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\correlation\\" + 'correlation.feather')
        self.X_train, self.X_hold, self.X_eval, self.X_test, self.X_train_oh, self.X_eval_oh, self.X_test_oh, self.y_train, \
            self.y_hold,  self.y_eval, self.y_test = split_dataset(self.features_correlated, self.target)
        #self.X_train_pca = pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\0.9\\" + 'X_train_pca.feather')
        #self.X_train_oh_pca = pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\0.9\\" + 'X_train_oh_pca.feather')
        #self.X_eval_pca = pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\0.9\\" + 'X_eval_pca.feather')
        #self.X_test_pca = pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\0.9\\" + 'X_test_pca.feather')

        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "/time_stats.txt", "a")
        

    def correlation_training(self) -> None:
        #self.time_stats_file.write("---- STARTING CORRELATION TRAINING ----")
        #self.time_stats_file.write("\n")
        #BaselineClassifier().train(self.X_train, self.y_train)
        #LogisticRegressionClassifier().train(self.X_train_oh, self.X_train, self.y_train)
        #RFClassifier().train(self.X_train_oh, self.X_train, self.y_train)
        #CatClassifier().train(self.X_eval, self.y_eval, self.X_train, self.y_train, self.X_test)
        DeepLearningClassifier(self.X_train, self.y_train, self.X_eval, self.y_eval, self.X_test, self.y_test).train()
        

    def PCA_training(self) -> None:
        pass
        #self.time_stats_file.write("---- STARTING PCA TRAINING ----")
        #self.time_stats_file.write("\n")
        #BaselineClassifier().train(self.X_train_pca, self.y_train)
        #LogisticRegressionClassifier().train(self.X_train_oh_pca, self.X_train_pca, self.y_train)
        #RFClassifier().train(self.X_train_oh_pca, self.X_train_pca, self.y_train)

    def pipeline(self) -> None:
        self.correlation_training()
        # self.PCA_training()
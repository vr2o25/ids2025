from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

from utils import save_model
from config.config import Config

class LogisticRegressionClassifier():
    def __init__(self) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "/time_stats.txt", "a")
        self.param_grid = [
                            {'penalty' : ['l1', 'l2'],
                            'C' : np.logspace(-4, 4, 8),
                            'solver' : ['saga']
                            }
                        ]


    def train(self, X_train_oh, X_train, y_train) -> None:
        print("Starting LogReg Classifier")
        self.start_time = time.time()
        
        scaler = StandardScaler()
        scaler.fit(X_train_oh)

        cls_lr = LogisticRegression(verbose=2)
        grid_model = GridSearchCV(estimator=cls_lr, param_grid=self.param_grid, cv = 5, verbose = 1)

        # cls_lr.fit(scaler.transform(X_train_oh), y_train.label_is_attack)
        grid_model.fit(scaler.transform(X_train_oh), y_train.label_is_attack)

        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Logistic Regression Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        save_model(grid_model, "logreg_model.sav")
        print("Saved LogReg Classifier")
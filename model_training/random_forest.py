from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

from utils import save_model
from config.config import Config

class RFClassifier():
    def __init__(self) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "/time_stats.txt", "a")

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        self.grid = {'max_features': max_features,
                    'bootstrap': bootstrap,
                    'class_weight': ['balanced'],
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

    def train(self, X_train_oh, X_train, y_train) -> None:
        print("Starting Random Forest Classifier")
        self.start_time = time.time()

        cls_forest = RandomForestClassifier(verbose=1, n_jobs=-1)

        #grid_model = GridSearchCV(estimator=cls_forest, param_grid=self.grid, cv = 5, verbose = 1, n_jobs = 10)
        grid_model = GridSearchCV(estimator=cls_forest, param_grid=self.grid, cv = 5, verbose = 1)
        

        # cls_forest.fit(X_train_oh, y_train.label_is_attack)
        grid_model.fit(X_train_oh, y_train.label_is_attack)

        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Random Forest Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        save_model(grid_model, "rf_model.sav")
        print("Saved Random Forest Classifier")
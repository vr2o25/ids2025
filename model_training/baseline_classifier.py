from sklearn.dummy import DummyClassifier
import time

from utils import save_model
from config.config import Config

class BaselineClassifier():
    def __init__(self) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "/time_stats.txt", "a")

    def train(self, X_train, y_train) -> None:
        print("Starting Baseline Classifier")
        self.start_time = time.time()
        cls_dummy = DummyClassifier('most_frequent')
        cls_dummy.fit(X_train, y_train.label_is_attack)
        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Baseline Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        save_model(cls_dummy, "baseline_model.sav")
        print("Saved Baseline Classifier")
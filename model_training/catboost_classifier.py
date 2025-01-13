from catboost import CatBoostClassifier
from catboost import Pool
import json

import time

from utils import save_model
from config.config import Config

class CatClassifier():
    def __init__(self) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "/time_stats.txt", "a")

    def train(self, X_eval, y_eval, X_train, y_train, X_test) -> None:
        print("Starting Catboost Classifier")
        self.start_time = time.time()

        train_pool = Pool(X_train, y_train.label_is_attack, cat_features=['protocol'])
        eval_pool = Pool(X_eval, y_eval.label_is_attack, cat_features=['protocol'])
        test_pool = Pool(X_test, cat_features=['protocol'])

        minority_class_weight = len(y_train[y_train.label_is_attack == 0]) / len(y_train[y_train.label_is_attack == 1])
            
        cls_cb = CatBoostClassifier(loss_function='Logloss',
                                    eval_metric='Recall',                        
                                    # class_weights=[1, minority_class_weight],
                                    task_type='GPU',
                                    verbose=True)

        grid = {'learning_rate': [0.03, 0.1],
                'depth': [4, 6, 10],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        
        grid_search_result = cls_cb.grid_search(grid,
                                       X=train_pool,
                                       y=eval_pool,
                                       cv=5)

        #cls_cb.fit(train_pool, eval_set=eval_pool)

        self.end_time = time.time()
        self.time_stats_file.write("---- Model stats for CatBoost Classifier ----")
        self.time_stats_file.write(json.dumps(grid_search_result.params))
        self.time_stats_file.write("---------------------------------------------")
        self.time_stats_file.write(json.dumps(grid_search_result.cv_results))
        self.time_stats_file.write("---- Time stats for CatBoost Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        save_model(cls_cb, "catboost_model.sav")
        print("Saved Catboost Classifier")
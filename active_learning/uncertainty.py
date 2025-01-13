import pandas as pd
import pyarrow.feather as feather
import time

from sklearn.metrics import log_loss
from typing import Dict, Optional, Tuple, List, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, metrics
import time




class Uncertainty:
    def __init__(self) -> None:
        """Pre-Loading datasets and federated model. Threshold definition"""
        self.THRESHOLD = 0.2
        self.PREDICT_BATCH_SIZE = 16384
        # load federated model
        self.model = models.load_model('E:\\Mestrado\\askonas-ids\\models\\federated\\3_clients\\50_rounds.h5')

        self.X_train = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier"+ '\\X_train.feather')
        self.y_train = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + '\\y_train.feather')
        self.X_val = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + '\\X_val.feather')
        self.y_val = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + "\\" + 'y_val.feather')
        self.X_test = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + '\\X_test.feather')
        self.y_test = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + "\\" + 'y_test.feather')

        # test sample the 2017 dataset
        self.X_train_2017 = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\2017" + '\\X_train.feather')
        self.y_train_2017 = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\2017" + "\\" + 'y_train.feather')
        self.X_test_2017 = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\2017" + '\\X_test.feather')
        self.y_test_2017 = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\2017" + "\\" + 'y_test.feather')
        self.X_val_2017 = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\2017" + '\\X_val.feather')
        self.y_val_2017 = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\2017" + "\\" + 'y_val.feather')


    def calculate_uncertainty(self, probability_array):
        """
        Calculates the uncertainty of a given probability distribution similar to
        array = [
                [0.9, 0.1],
                [0.8, 0.2]
        ]
        """
        return 1 - probability_array.max(axis=1)

    def select_samples(self, X, array):
        """Selects the most uncertain samples by appending the value 1 to the a new column in the dataframe, 0 otherwise."""
        bin_uncertainty = []

        probability_array = self.calculate_uncertainty(array)

        for item in probability_array:
            if item > self.THRESHOLD:
                bin_uncertainty.append(1)
            else:
                bin_uncertainty.append(0)


        X[30] = bin_uncertainty

        return X        

    def reform_dataset(self, X, y, original_X, original_y):
        """Prepares a new addition to be appended to the original dataset"""
        if 'label_is_attack' not in y.columns:
            y = y.set_axis([*y.columns[:-1], 'label_cat'], axis=1, inplace=False)
            y['label_is_attack'] = y['label_cat'].apply(lambda x: 0 if x == 0 else 1)
        if 'label_is_attack' not in original_y.columns:
            original_y = original_y.set_axis([*original_y.columns[:-1], 'label_cat'], axis=1, inplace=False)
            original_y['label_is_attack'] = original_y['label_cat'].apply(lambda x: 0 if x == 0 else 1)
        result = pd.concat([X, y], axis=1)
        data = []

        for _, row in result.iterrows():
            if row[30] == 1:
                data.append(list(row))

        df = pd.DataFrame(data, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 'label_cat', 'label_is_attack'])

        new_X = df.drop(columns=[30, 'label_cat', 'label_is_attack'])
        new_X.columns = new_X.columns.astype(str)
        new_y = df[['label_is_attack', 'label_cat']]
        enhanced_X = pd.concat([original_X, new_X], ignore_index=True, sort=False)
        enhanced_y = pd.concat([original_y, new_y], ignore_index=True, sort=False)
        self.serialize_dataset(enhanced_X, "enhanced_X_train_2.feather")
        self.serialize_dataset(enhanced_y, "enhanced_y_train_2.feather")
        return enhanced_X, enhanced_y

    def prepare_probability_distribution(self, X):
        """
        Uses the tf.model.predict method to calculate the resulting prediction probability. 
        Then, a probability distribution is calculated by subtracting the prediction probability by 1. 
        """
        prob_distribution = self.model.predict(X)
        a = prob_distribution.reshape((len(prob_distribution), ))
        final_array = np.zeros(shape=(len(prob_distribution),2))
        for i in range(len(a)):
            final_array[i] = [a[i], 1-a[i]]

        return final_array

    def serialize_dataset(self, df, file_name) -> None:
        print("Serializing at: ", "E:\\Mestrado\\askonas-ids\\federated\\datasets" + "\\" + file_name)
        feather.write_feather(df, "E:\\Mestrado\\askonas-ids\\federated\\datasets" + "\\" + file_name)
    
    def pipeline(self):
        """Performs all methods needed for uncertainty caculation and dataset generation."""
        array = self.prepare_probability_distribution(self.X_train_2017)
        test = self.select_samples(self.X_train_2017, array)        
        self.reform_dataset(test, self.y_train_2017, self.X_train, self.y_train)
        #print(enhanced_x, enhanced_y)

if __name__ == "__main__":
    Uncertainty().pipeline()
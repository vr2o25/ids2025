from sklearn.model_selection import train_test_split
import pandas as pd
import gc
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError
import pickle
from typing import Tuple, Union, List
from collections import Counter
import os
import glob

from config.config import Config

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def save_model(model, filename):
    pickle.dump(model, open(Config.MODELS_FOLDER + "/" + filename, 'wb'))

def print_report(ds_type, cls, X_vals, y_true, y_predict, plot_pr=False, plot_roc=False):
    print(f"Classification Report ({ds_type}):")
    print(classification_report(y_true, y_predict))
    print(f"Avg Precision Score: {average_precision_score(y_true, y_predict, average='weighted')}")
    
    if plot_roc:
        print(f"ROC AUC Score: {roc_auc_score(y_true, y_predict)}")
        skplt.metrics.plot_roc(y_true, cls.predict_proba(X_vals))
        plt.show()
        
    if plot_pr:
        
        skplt.metrics.plot_precision_recall(y_true, cls.predict_proba(X_vals))
        plt.show()
        
    print('\n')
    
def split_dataset(X, y):
  X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, stratify=y['label_cat'], random_state=42)
  X_eval, X_test, y_eval, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold['label_cat'], random_state=42)
  
  X_train_oh = pd.get_dummies(X_train, columns=['protocol'])
  X_eval_oh = pd.get_dummies(X_eval, columns=['protocol'])
  X_test_oh = pd.get_dummies(X_test, columns=['protocol'])
  
  return X_train, X_hold, X_eval, X_test, X_train_oh, X_eval_oh, X_test_oh, y_train, \
            y_hold,  y_eval, y_test




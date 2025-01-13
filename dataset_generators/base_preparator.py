import numpy as np
import pandas as pd
import glob
import os
import pyarrow.feather as feather

from config.config import Config

# TODO: Logging

class BasePreparator():
    def __init__(self) -> None:
        print(Config.CIC_IDS_2018_PROCESSED_CSVS)
        self.csv_files = glob.glob(os.path.join(Config.CIC_IDS_2018_PROCESSED_CSVS, '*.csv'))
        self.df: pd.Dataframe = pd.concat((pd.read_csv(f, dtype=Config.data_types) for f in self.csv_files))
        self.features: pd.DataFrame
        self.target: pd.Dataframe

    def replace_inf_in_columns(self) -> None:
        print("replacing infs")
        inf_columns = [c for c in self.df.columns if self.df[self.df[c] == np.inf][c].count() > 0]
        for col in inf_columns:
            self.df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            mean = self.df[col].mean()
            self.df[col].fillna(mean, inplace=True)

    def replace_negative_values_with_mean(self) -> None:
        print("replacing numerics")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.values
        
        columns = [c for c in numeric_cols if self.df[self.df[c] < 0][c].count() > 0]
        for col in columns:
            mask = self.df[col] < 0
            self.df.loc[mask, col] = np.nan
            mean = self.df[col].mean()
            self.df[col].fillna(mean, inplace=True)

    def create_labels(self) -> None:
        print("creating labels")
        self.df['label'] = self.df.label.astype('category')
        self.df['label_code'] = self.df['label'].cat.codes
        self.df['label_is_attack'] = self.df.label.apply(lambda x: 0 if x == 'Benign' else 1)

        attack_types = [a for a in self.df.label.value_counts().index.tolist() if a != 'Benign']

        for a in attack_types:
            l = 'label_is_attack_' + a.replace('-', ' ').replace(' ', '_').lower()
            self.df[l] = self.df.label.apply(lambda x: 1 if x == a else 0)
        
        self.df['label_cat'] = self.df.label.astype('category').cat.codes
        self.df['label_is_attack'] = (self.df.label != 'Benign').astype('int')
    
    def serialize_dataset(self, df, file_name) -> None:
        print("Serializing at: ", Config.DATASETS_FOLDER + "/" + file_name)
        feather.write_feather(df, Config.DATASETS_FOLDER + "/" + file_name)

    def drop_undesired_features(self) -> None:
        print("dropping features")
        stats = self.df.describe()
        std = stats.loc['std']
        features_no_variance = std[std == 0.0].index
        self.df = self.df.drop(columns=features_no_variance)  # dropping features with no variance
        
        self.df = self.df.drop(columns=['timestamp', 'dst_port'])  # dropping features that will not be relevant to final estimation
    
    def drop_minority_class_rows(self):
        print("Dropping minority classes")
        self.df.drop(self.df.index[self.df['label'] == 'SQL Injection'], inplace=True)
        self.df.drop(self.df.index[self.df['label'] == 'Brute Force -XSS'], inplace=True)
        self.df.drop(self.df.index[self.df['label'] == 'Brute Force -Web'], inplace=True)
        self.df.drop(self.df.index[self.df['label'] == 'DDOS attack-LOIC-UDP'], inplace=True)
        self.df.drop(self.df.index[self.df['label'] == 'DoS attacks-Slowloris'], inplace=True)
        self.df.drop(self.df.index[self.df['label'] == 'DoS attacks-GoldenEye'], inplace=True)
    
    def feature_target_separation(self) -> None:
        print("splitting")
        #self.features = df.drop(columns=['label', 'label_cat', 'label_is_attack'])
        self.features = self.df[self.df.columns.drop(list(self.df.filter(regex='label')))]
        self.target = self.df[['label_is_attack', 'label_cat', 'label']]

    def base_dataset_pipeline(self):
        self.replace_inf_in_columns()
        self.replace_negative_values_with_mean()
        self.create_labels()
        self.drop_undesired_features()
        self.drop_minority_class_rows()
        self.feature_target_separation()
        self.serialize_dataset(self.features, 'features_dataset.feather')
        self.serialize_dataset(self.target, 'target_dataset.feather')

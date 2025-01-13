import numpy as np
import pandas as pd
import glob
import os
import pyarrow.feather as feather
#from imblearn.over_sampling import SMOTENC
#from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter

from config.config import Config

class Balancer():
    def __init__(self) -> None:
        self.features: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "/" + 'features_dataset.feather')
        self.target: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "/" + 'target_dataset.feather')
        
        self.types = {
          'protocol': 'uint8',
          'flow_duration': 'int64',
          'tot_fwd_pkts': 'uint32',
          'tot_bwd_pkts': 'uint32',
          'totlen_fwd_pkts': 'uint32',
          'totlen_bwd_pkts': 'uint32',
          'fwd_pkt_len_max': 'uint16',
          'fwd_pkt_len_min': 'uint16',
          'fwd_pkt_len_mean': 'float32',
          'fwd_pkt_len_std': 'float32',
          'bwd_pkt_len_max': 'uint16',
          'bwd_pkt_len_min': 'uint16',
          'bwd_pkt_len_mean': 'float32',
          'bwd_pkt_len_std': 'float32',
          'flow_byts_s': 'float64',
          'flow_pkts_s': 'float64',
          'flow_iat_mean': 'float32',
          'flow_iat_std': 'float32',
          'flow_iat_max': 'int64',
          'flow_iat_min': 'int64',
          'fwd_iat_tot': 'int64',
          'fwd_iat_mean': 'float32',
          'fwd_iat_std': 'float32',
          'fwd_iat_max': 'int64',
          'fwd_iat_min': 'int64',
          'bwd_iat_tot': 'uint32',
          'bwd_iat_mean': 'float32',
          'bwd_iat_std': 'float32',
          'bwd_iat_max': 'uint32',
          'bwd_iat_min': 'uint32',
          'fwd_psh_flags': 'uint8',
          'fwd_urg_flags': 'uint8',
          'fwd_header_len': 'uint32',
          'bwd_header_len': 'uint32',
          'fwd_pkts_s': 'float32',
          'bwd_pkts_s': 'float32',
          'pkt_len_min': 'uint16',
          'pkt_len_max': 'uint16',
          'pkt_len_mean': 'float32',
          'pkt_len_std': 'float32',
          'pkt_len_var': 'float32',
          'fin_flag_cnt': 'uint8',
          'syn_flag_cnt': 'uint8',
          'rst_flag_cnt': 'uint8',
          'psh_flag_cnt': 'uint8',
          'ack_flag_cnt': 'uint8',
          'urg_flag_cnt': 'uint8',
          'cwe_flag_count': 'uint8',
          'ece_flag_cnt': 'uint8',
          'down_up_ratio': 'uint16',
          'pkt_size_avg': 'float32',
          'fwd_seg_size_avg': 'float32',
          'bwd_seg_size_avg': 'float32',
          'subflow_fwd_pkts': 'uint32',
          'subflow_fwd_byts': 'uint32',
          'subflow_bwd_pkts': 'uint32',
          'subflow_bwd_byts': 'uint32',
          'init_fwd_win_byts': 'int32',
          'init_bwd_win_byts': 'int32',
          'fwd_act_data_pkts': 'uint32',
          'fwd_seg_size_min': 'uint8',
          'active_mean': 'float32',
          'active_std': 'float32',
          'active_max': 'uint32',
          'active_min': 'uint32',
          'idle_mean': 'float32',
          'idle_std': 'float32',
          'idle_max': 'uint64',
          'idle_min': 'uint64',
      }
    
    def serialize_dataset(self, df, file_name) -> None:
        print("Serializing at: ", Config.DATASETS_FOLDER + "/" + file_name)
        feather.write_feather(df, Config.DATASETS_FOLDER + "/" + file_name)
        

    def undersamppler(self):
        print("Undersampling")
        print(self.features)
        cnts = self.target.label_cat.value_counts()
        sample_dict = {}

        for i in np.unique(self.target.label_cat):
            sample_dict[i] = max(cnts[i], 87468) 
        rus = NearMiss(version=1, n_neighbors=3)
        #X_train_undersampled, y_train_undersampled = undersample.fit_resample(self.features, self.target.label_cat)
        #rus = RandomUnderSampler(sampling_strategy=sample_dict, random_state=42, replacement=True)
        X_train_undersampled, y_train_undersampled = rus.fit_resample(self.features, self.target.label_cat)

        features = pd.DataFrame(X_train_undersampled, columns=self.features.columns)
        
        for key, value in self.types.items():
          features[key] = features[key].astype(value)
        
        target = pd.DataFrame(y_train_undersampled, columns=["label_cat"])
        
        target['label_is_attack'] = target.label_cat.apply(lambda x: 0 if x == 0 else 1)
        
        print(features)
        print(target)

        self.serialize_dataset(features, 'balanced/undersample_features_dataset.feather')
        self.serialize_dataset(target, 'balanced/undersample_target_dataset.feather')

    def oversampler(self):
        print("Oversampling")
        cnts = self.target.label_cat.value_counts()
        sample_dict = {}

        for i in np.unique(self.target.label_cat):
            sample_dict[i] = max(cnts[i], 10000000)        

        sm = SMOTENC(sampling_strategy=sample_dict, categorical_features=[0], n_jobs=24, random_state=42)
        X_train_oversampled, y_train_oversampled = sm.fit_resample(self.features, self.target.label_cat)

        features = pd.DataFrame(X_train_oversampled, columns=self.features.columns)
        target = pd.DataFrame(y_train_oversampled, columns=self.target.label_cat)

        self.serialize_dataset(features, 'balanced/oversample_features_dataset.feather')
        self.serialize_dataset(target, 'balanced/oversample_target_dataset.feather')

    def sampler(self):
        #self.oversampler()
        self.undersamppler()





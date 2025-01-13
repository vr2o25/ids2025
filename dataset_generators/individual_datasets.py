from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pyarrow.feather as feather

from config.config import Config
from utils import *

class IndividualDatasetPrep():
    def __init__(self) -> None:
        self.features: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'features_dataset.feather')
        self.target: pd.DataFrame = feather.read_feather(Config.DATASETS_FOLDER + "\\" + 'target_dataset.feather')
        self.target = self.target.replace('Infilteration', 'Infiltration')
        self.df: pd.DataFrame = pd.concat([self.features, self.target], axis=1)

    def serialize_dataset(self, df, file_name) -> None:
        print("Serializing at: ", Config.DATASETS_FOLDER + "\\individual_datasets\\" + file_name)
        feather.write_feather(df, Config.DATASETS_FOLDER + "\\individual_datasets\\" + file_name)

    def pipeline(self):
        self.df = self.df[~self.df.label.str.contains("DoS attacks-GoldenEye")].reset_index(drop=True)
        self.df = self.df[~self.df.label.str.contains("DoS attacks-Slowloris")].reset_index(drop=True)
        self.df = self.df[~self.df.label.str.contains("DDOS attack-LOIC-UDP")].reset_index(drop=True)
        self.df = self.df[~self.df.label.str.contains("Brute Force -Web")].reset_index(drop=True)
        self.df = self.df[~self.df.label.str.contains("Brute Force -XSS")].reset_index(drop=True)
        self.df = self.df[~self.df.label.str.contains("SQL Injection")].reset_index(drop=True)
        
        ddos_hoic = self.df.loc[self.df['label'].isin(['DDOS attack-HOIC', 'Benign'])]
        ddos_loic_http = self.df.loc[self.df['label'].isin(['DDoS attacks-LOIC-HTTP', 'Benign'])]
        dos_hulk = self.df.loc[self.df['label'].isin(['DoS attacks-Hulk', 'Benign'])]
        bot = self.df.loc[self.df['label'].isin(['Bot', 'Benign'])]
        ftp_bruteforce = self.df.loc[self.df['label'].isin(['FTP-BruteForce', 'Benign'])]
        ssh_bruteforce = self.df.loc[self.df['label'].isin(['SSH-BruteForce', 'Benign'])]
        infiltration = self.df.loc[self.df['label'].isin(['Infiltration', 'Benign'])]
        
        self.serialize_dataset(ddos_hoic, 'ddos_hoic.feather')
        self.serialize_dataset(ddos_loic_http, 'ddos_loic_http.feather')
        self.serialize_dataset(dos_hulk, 'dos_hulk.feather')
        self.serialize_dataset(bot, 'bot.feather')
        self.serialize_dataset(ftp_bruteforce, 'ftp_bruteforce.feather')
        self.serialize_dataset(ssh_bruteforce, 'ssh_bruteforce.feather')
        self.serialize_dataset(infiltration, 'infiltration.feather')



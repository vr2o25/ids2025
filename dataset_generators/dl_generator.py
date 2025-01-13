from sklearn.model_selection import train_test_split
import pandas as pd
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError
from imblearn.over_sampling import SMOTE, SMOTENC
from typing import Tuple, List
from collections import Counter
import os
import glob
import pyarrow.feather as feather
from numpy import asarray
from numpy import save

from config.config import Config

def serialize_dataset() -> None:

    dataset = load_dataset(Config.CIC_IDS_2018_PROCESSED_CSVS,
                        omit_cols=Config.FEATURES_NO_VARIANCE + ['timestamp', 'dst_port', 'protocol'],
                        preserve_neg_value_cols=['init_fwd_win_byts', 'init_bwd_win_byts'])

    X_train, y_train, X_val, y_val, X_test, y_test, column_names = transform_data(dataset=dataset,
                                                                                imputer_strategy='median',
                                                                                scaler=StandardScaler,
                                                                                attack_samples=100000,
                                                                               random_state=Config.rand_state)
    print(type(column_names))
    print(column_names)

    print("Serializing at: ", Config.DATASETS_FOLDER + "\\federated\\")
    save(Config.DATASETS_FOLDER + "\\federated\\" + "X_train.npy", X_train, allow_pickle=True)
    save(Config.DATASETS_FOLDER + "\\federated\\" + "y_train.npy", y_train, allow_pickle=True)
    save(Config.DATASETS_FOLDER + "\\federated\\" + "X_val.npy", X_val, allow_pickle=True)
    feather.write_feather(y_val, Config.DATASETS_FOLDER + "\\federated\\" + 'y_val.feather')
    #save(Config.DATASETS_FOLDER + "\\federated\\" + "y_val.npy", y_val, allow_pickle=True)
    save(Config.DATASETS_FOLDER + "\\federated\\" + "X_test.npy", X_test, allow_pickle=True)
    feather.write_feather(y_test, Config.DATASETS_FOLDER + "\\federated\\" + 'y_test.feather')
    #save(Config.DATASETS_FOLDER + "\\federated\\" + "y_test.npy", y_test, allow_pickle=True)
    save(Config.DATASETS_FOLDER + "\\federated\\" + "column_names.npy", column_names, allow_pickle=True)

def split_dataset(X, y):

        X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, stratify=y.label_cat, random_state=42)
        X_eval, X_test, y_eval, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold.label_cat, random_state=42)

        X_train_oh = pd.get_dummies(X_train, columns=['protocol'])
        X_eval_oh = pd.get_dummies(X_eval, columns=['protocol'])
        X_test_oh = pd.get_dummies(X_test, columns=['protocol'])

        return X_train, X_hold, X_eval, X_test, X_train_oh, X_eval_oh, X_test_oh, y_train, y_hold,  y_eval, y_test

def train_val_test_split(df: pd.DataFrame,
                         val_size: float = 0.1,
                         test_size: float = 0.1,
                         stratify_col: str = None,
                         random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the given DataFrame into three parts used for:
    - training
    - validation
    - test
    :param df: Input DataFrame.
    :param val_size: Size of validation set.
    :param test_size: Size of test set.
    :param stratify_col: Column to stratify.
    :param random_state: Random state.
    :return: A triple containing (`train`, `val`, `test`) sets.
    """
    assert (val_size + test_size) < 1, 'Sum of validation and test size must not be > 1.'

    df_stratify = df[stratify_col] if stratify_col else None
    df_train, df_hold = train_test_split(df,
                                         test_size=(val_size + test_size),
                                         stratify=df_stratify,
                                         random_state=random_state)

    df_hold_stratify = df_hold[stratify_col] if stratify_col else None
    df_val, df_test = train_test_split(df_hold,
                                       test_size=test_size / (val_size + test_size),
                                       stratify=df_hold_stratify,
                                       random_state=random_state)

    return df_train, df_val, df_test

def split_x_y(df: pd.DataFrame, y_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the given DataFrame into a DataFrame `X` containing the predictor variables and a DataFrame 'y' containing
    the labels y.
    :param df: Input DataFrame.
    :param y_cols: Columns to use in the labels DataFrame `y`.
    :return: A tuple containing the DataFrames (`X`, `y`).
    """
    if y_cols is None:
        y_cols = ['label', 'label_cat', 'label_is_attack']
    return df.drop(columns=y_cols), df[y_cols]

def upsample_minority_classes(X: np.ndarray,
                              y: pd.DataFrame,
                              min_samples: int,
                              random_state: int = None,
                              cat_cols: List[int] = None,
                              n_jobs: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic up-sampling of minority classes using `imblearn.over_sampling.SMOTE`.
    :param X: Predictor variables.
    :param y: Labels.
    :param min_samples: Minimum samples of each class.
    :param random_state: Random state.
    :param cat_cols: Column indices of categorical features.
    :param n_jobs: Number of threads to use.
    :return: A tuple containing the up-sampled X and y values.
    """
    counts = y.label_cat.value_counts()
    sample_dict = {}

    for i in np.unique(y.label_cat):
        sample_dict[i] = max(counts[i], min_samples)

    if cat_cols:
        smote = SMOTENC(sampling_strategy=sample_dict,
                        categorical_features=cat_cols,
                        n_jobs=n_jobs,
                        random_state=random_state)
    else:
        smote = SMOTE(sampling_strategy=sample_dict, n_jobs=n_jobs, random_state=random_state)

    x_s, y_s = smote.fit_resample(X, y.label_cat)
    return x_s, y_s


def create_sample_dict(df: pd.DataFrame,
                       default_nr_samples: int,
                       samples_per_label: dict = None) -> dict:
    """
    Creates a dictionary containing the number of samples per label.
    :param df: Input DataFrame.
    :param default_nr_samples: Default number of samples per label.
    :param samples_per_label: Number of samples for specific labels.
    :return: Dictionary containing the number of samples per label.
    """
    if samples_per_label is None:
        samples_per_label = {}

    sample_dict = df.label_cat.value_counts().to_dict()

    for label in sample_dict.keys():
        requested_samples = samples_per_label[label] if label in samples_per_label else default_nr_samples
        existing_samples = sample_dict[label] if label in sample_dict else 0
        sample_dict[label] = min(requested_samples, existing_samples)

    return sample_dict


def downsample(df: pd.DataFrame,
               default_nr_samples: int,
               samples_per_label: dict = None,
               random_state: int = None) -> pd.DataFrame:
    """
    Downsamples the given DataFrame to contain at most `default_nr_samples` per instance of label.
    :param df: Input DataFrame.
    :param default_nr_samples: Default number of samples per label.
    :param samples_per_label: Number of samples for specific labels.
    :param random_state: Random state.
    :return: The downsampled DataFrame.
    """
    if samples_per_label is None:
        samples_per_label = {}

    sample_dict = create_sample_dict(df, default_nr_samples, samples_per_label)
    return pd.concat([df[df.label_cat == l].sample(n=n, random_state=random_state) for l, n in sample_dict.items()])

def create_pipeline(df: pd.DataFrame,
                    imputer_strategy: str = 'mean',
                    imputer_cols: List[str] = None,
                    scaler: BaseEstimator = StandardScaler,
                    scaler_args: dict = None,
                    cat_cols: List[str] = None,
                    copy: bool = True):
    """
    Creates a pipeline performing the following steps:
    - value imputation
    - value scaling
    - one-hot-encoding of categorical values.
    :param df: Input DataFrame.
    :param imputer_strategy: Imputer strategy applied to missing values.
                             Allowed values are ['mean', 'median', 'most_frequent', 'constant'].
    :param imputer_cols: Columns to impute. If no columns are specified all columns will be imputed.
    :param scaler: Scikit-learn scaler to be applied to all values.
    :param scaler_args: Additional arguments forwarded to the specified scaler.
    :param cat_cols: Categorical columns to be one-hot-encoded.
    :param copy: If True, a copy of the input will be created.
    :return: A tuple containing the pipeline and a function returning the columns names after the pipeline has been
             fitted.
    """

    def create_get_feature_names(p, imp, scl, cat):
        def get_feature_names():
            if not hasattr(p, 'transformers_'):
                raise AssertionError('Pipeline is not yet fitted.')

            try:
                cat_names = p.transformers_[2][1].get_feature_names(cat)
            except NotFittedError:
                cat_names = []
            return np.append(imp, np.append(scl, cat_names))

        return get_feature_names

    if scaler_args is None:
        scaler_args = {}

    cat_features = cat_cols if cat_cols else []
    num_features = [c for c in df.select_dtypes(include=[np.number]).columns.values if c not in cat_features]
    imp_features: List[str] = []

    if imputer_strategy is not None:
        imp_features = imputer_cols if imputer_cols else num_features

    scale_features = [f for f in num_features if f not in imp_features]

    imp_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy=imputer_strategy, copy=copy)),
        ('imp_scaler', scaler(**scaler_args))
    ])

    pipeline = ColumnTransformer([
        ('imp', imp_pipeline, imp_features),
        ('scl', scaler(**scaler_args), scale_features),
        ('one_hot', OneHotEncoder(categories='auto'), cat_features)
    ])

    return pipeline, create_get_feature_names(pipeline, imp_features, scale_features, cat_features)

def transform_data(dataset,
                   attack_samples,
                   imputer_strategy,
                   scaler,
                   benign_samples=None,
                   random_state=None):

    cols_to_impute = dataset.columns[dataset.isna().any()].tolist()

    train_data, val_data, test_data = train_val_test_split(dataset,
                                                           val_size=0.1,
                                                           test_size=0.1,
                                                           stratify_col='label_cat',
                                                           random_state=random_state)

    if benign_samples:
        train_data = downsample(train_data, default_nr_samples=benign_samples, random_state=random_state)

    X_train_raw, y_train = split_x_y(train_data)
    X_val_raw, y_val = split_x_y(val_data)
    X_test_raw, y_test = split_x_y(test_data)

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train_raw.shape))
    print('Val:      {}'.format(X_val_raw.shape))
    print('Test:     {}'.format(X_test_raw.shape))

    print('\nTraining labels:')
    print('================')
    print(y_train.label.value_counts())
    print('\nValidation labels:')
    print('==================')
    print(y_val.label.value_counts())
    print('\nTest labels:')
    print('============')
    print(y_test.label.value_counts())

    del train_data, val_data, test_data
    gc.collect()

    pipeline, get_col_names = create_pipeline(X_train_raw,
                                              imputer_strategy=imputer_strategy,
                                              imputer_cols=cols_to_impute,
                                              scaler=scaler)

    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    column_names = get_col_names()

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train.shape))
    print('Val:      {}'.format(X_val.shape))
    print('Test:     {}'.format(X_test.shape))

    print('\nMissing values:')
    print('===============')
    print('Training: {}'.format(np.count_nonzero(np.isnan(X_train))))
    print('Val:      {}'.format(np.count_nonzero(np.isnan(X_val))))
    print('Test:     {}'.format(np.count_nonzero(np.isnan(X_test))))

    print('\nScaling:')
    print('========')
    print('Training: min={}, max={}'.format(np.min(X_train), np.max(X_train)))
    print('Val:      min={}, max={}'.format(np.min(X_val), np.max(X_val)))
    print('Test:     min={}, max={}'.format(np.min(X_test), np.max(X_test)))

    X_train, y_train = upsample_minority_classes(X_train,
                                                 y_train,
                                                 min_samples=attack_samples,
                                                 random_state=random_state)

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train.shape))

    print('\nTraining labels:')
    print('================')
    print(Counter(y_train))

    return X_train, y_train, X_val, y_val, X_test, y_test, column_names

def remove_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces values of type `np.inf` and `-np.inf` in a DataFrame with `null` values.
    :param df: Input DataFrame.
    :return: The DataFrame without `np.inf` and `-np.inf` values.
    """
    inf_columns = [c for c in df.columns if df[df[c] == np.inf][c].count() > 0]
    for col in inf_columns:
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def remove_negative_values(df: pd.DataFrame, ignore_cols: List[str] = None) -> pd.DataFrame:
    """
    Removes negative values in a DataFrame with `null` values.
    :param df: Input DataFrame.
    :param ignore_cols: Columns to ignore. Negative values in this columns will be preserved.
    :return: The DataFrame without negative values.
    """
    if ignore_cols is None:
        ignore_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(ignore_cols).values

    columns = [c for c in numeric_cols if df[df[c] < 0][c].count() > 0]
    for col in columns:
        mask = df[col] < 0
        df.loc[mask, col] = np.nan
    return df


def add_label_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column `label_cat` to the DataFrame specifying the category of the label.
    :param df: Input DataFrame.
    :return: The DataFrame containing a new column `label_cat`.
    """
    df[Config.COLUMN_LABEL_CAT] = df.label.apply(lambda l: Config.LABEL_CAT_MAPPING[l])
    return df


def add_label_is_attack_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column `label_is_attack` to the DataFrame containing a binary indicator specifying if a row is of category
    `benign = 0` or `attack = 1`.
    :param df: Input DataFrame.
    :return: The DataFrame containing a new column `label_is_attack`.
    """
    df[Config.COLUMN_LABEL_IS_ATTACK] = df.label.apply(lambda l: 0 if l == Config.LABEL_BENIGN else 1)
    return df

def load_dataset_generic(load_df_fn,
                         dataset_path: str,
                         use_cols: List[str] = None,
                         omit_cols: List[str] = None,
                         preserve_neg_value_cols: list = None,
                         transform_data: bool = True) -> pd.DataFrame:
    """
    Loads the dataset from the given path using the supplied function.
    All invalid values (`np.inf`, `-np.inf`, negative) are removed and replaced with `null` for easy imputation.
    Negative values of columns specified in `preserve_neg_value_cols` will be preserved.
    :param load_df_fn: Function used to load the dataset.
    :param dataset_path: Path of the base directory containing all files of the dataset.
    :param use_cols: Columns to load.
    :param omit_cols: Columns to omit.
    :param nrows: Number of rows to load per file.
    :param transform_data: Indicates if data should be manipulated (removal of invalid and negative values).
    :param preserve_neg_value_cols: Columns in which negative values are preserved.
    :return: The dataset as a DataFrame.
    """
    cols = None
    if use_cols:
        cols = use_cols
    if omit_cols:
        cols = [c for c in Config.data_types.keys() if c not in omit_cols]

    df = load_df_fn(dataset_path, cols)

    if transform_data:
        df = remove_inf_values(df)
        df = remove_negative_values(df, preserve_neg_value_cols)

    if Config.COLUMN_LABEL in df.columns:
        df = add_label_category_column(df)
        df = add_label_is_attack_columns(df)

    return df

def load_dataset(dataset_path: str,
                 use_cols: List[str] = None,
                 omit_cols: List[str] = None,
                 nrows: int = None,
                 transform_data: bool = True,
                 preserve_neg_value_cols: list = None) -> pd.DataFrame:
    """
    Loads the dataset in CSV format from the given path.
    All invalid values (`np.inf`, `-np.inf`, negative) are removed and replaced with `null` for easy imputation.
    Negative values of columns specified in `preserve_neg_value_cols` will be preserved.
    :param dataset_path: Path of the base directory containing all files of the dataset.
    :param use_cols: Columns to load.
    :param omit_cols: Columns to omit.
    :param nrows: Number of rows to load per file.
    :param transform_data: Indicates if data should be manipulated (removal of invalid and negative values).
    :param preserve_neg_value_cols: Columns in which negative values are preserved.
    :return: The dataset as a DataFrame.
    """

    def load_csv(path, cols):
        files = glob.glob(os.path.join(path, '*.csv'))
        return pd.concat([pd.read_csv(f, dtype=Config.data_types, usecols=cols, nrows=nrows) for f in files])

    return load_dataset_generic(load_df_fn=load_csv,
                                dataset_path=dataset_path,
                                use_cols=use_cols,
                                omit_cols=omit_cols,
                                preserve_neg_value_cols=preserve_neg_value_cols,
                                transform_data=transform_data)
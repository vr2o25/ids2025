import pyarrow.feather as feather
import numpy as np
import time

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import uuid

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, metrics
from collections import Counter
from hyperopt import fmin, hp, tpe, atpe, Trials, STATUS_OK

from model_training.dl_utils.data.dataset import load_dataset
from model_training.dl_utils.data.metadata import FEATURES_NO_VARIANCE
from model_training.dl_utils.utils import transform_data

from config.config import Config
from utils import *


class DeepLearningClassifier():
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test) -> None:
        self.start_time: float
        self.end_time: float
        self.time_stats_file = open(Config.STATS_AND_IMAGES_FOLDER + "/time_stats.txt", "a")
        self.rand_state = Config.rand_state
        tf_random = tf.random.set_seed(self.rand_state)
        np_random = np.random.seed(self.rand_state)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        # self.dataset = load_dataset(Config.CIC_IDS_2018_PROCESSED_CSVS,
        #                omit_cols=FEATURES_NO_VARIANCE + ['timestamp', 'dst_port', 'protocol'],
        #                preserve_neg_value_cols=['init_fwd_win_byts', 'init_bwd_win_byts'])

        # self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.column_names = transform_data(dataset=self.dataset,
        #     imputer_strategy='median', scaler=StandardScaler, attack_samples=100000, random_state=self.rand_state)

        

        y_train_is_attack = (self.y_train.label_is_attack != 0).astype('int')

        minority_class_weight = len(y_train_is_attack[y_train_is_attack == 0]) / len(y_train_is_attack[y_train_is_attack == 1])

        self.class_weights = { 
            0: 1, 
            1: minority_class_weight
        }    

    def create_model(self, input_dims, nr_layers, nr_units, activation, kerner_initializer, optimizer, dropout_layer=None):
        model = models.Sequential()
        model.add(layers.Input(shape=[input_dims]))
        
        for l in range(nr_layers):
            model.add(layers.Dense(nr_units, activation=activation, kernel_initializer=kerner_initializer))
            
        if dropout_layer:
            model.add(dropout_layer)
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=optimizer, 
                    loss='binary_crossentropy', 
                    metrics=[metrics.AUC(curve='PR'),
                            metrics.Precision(), 
                            metrics.Recall()])
        return model

    def train_best_parameter_model(self, args):
        print('\nRun') 
        print('==========')
        print('Parameters:\n{}'.format(args))
        
        epochs = int(args['epochs'])
        batch_size = int(args['batch_size'])
        nr_layers = int(args['nr_layers'])
        nr_units = int(args['nr_units'])
        activation = args['activation']
        dropout_rate = args['dropout_rate']
        model_name = 'models/{}.h5'.format(uuid.uuid4())
        
        if activation == 'elu':
            kerner_initializer = 'he_normal'
            dropout_layer = layers.Dropout(dropout_rate)
        elif activation == 'selu':
            kerner_initializer = 'lecun_normal'
            dropout_layer = layers.AlphaDropout(dropout_rate)
        else:
            raise ValueError('Invalid activation "{}" supplied.'.format(opt_args['name']))
        
        opt_name = args['optimizer']['name']
        opt_lr_mult = args['optimizer']['lr_mult']
        
        if opt_name == 'sgd':
            optimizer = optimizers.SGD(lr=(0.01 * opt_lr_mult), momentum=0.9, nesterov=True)
        elif opt_name == 'adam':
            optimizer = optimizers.Adam(lr=(0.001 * opt_lr_mult))
        elif opt_name == 'nadam':
            optimizer = optimizers.Nadam(lr=(0.002 * opt_lr_mult))
        else:
            raise ValueError('Invalid optimizer "{}" supplied.'.format(opt_args['name']))
            
        #K.clear_session()
        #gc.collect()
        
        model = self.create_model(input_dims=self.X_train.shape[1], 
                            nr_layers=nr_layers, 
                            nr_units=nr_units, 
                            activation=activation, 
                            kerner_initializer=kerner_initializer,
                            optimizer=optimizer,
                            dropout_layer=dropout_layer)
        
        mc = callbacks.ModelCheckpoint(filepath=model_name, save_best_only=True, verbose=0)
        
        lr_scheduler = callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        
        hist = model.fit(x=self.X_train, 
                        y=self.y_train_is_attack, 
                        validation_data=(self.X_val, self.y_val.label_is_attack.values),
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=self.class_weights,
                        callbacks=[
                            lr_scheduler,
                            mc
                        ],
                        verbose=2)
        
        best_loss = np.amin(hist.history['val_loss']) 
        print('Best loss: {}'.format(best_loss))
        print('Model: {}'.format(model_name))
        
        return {
            'loss': best_loss,
            'status': STATUS_OK,
            'model_name': model_name
        }    

    def train_optimized_model(self, X_train, y_train, X_val, y_val, model_path, epochs, batch_size, class_weights):    
        input_dims = X_train.shape[1]

        #K.clear_session()
        #gc.collect()
        
        nr_layers = 5
        nr_units = 300
        dropout_rate = 0.22339774943469998
        lr = (0.001 * 0.61157158868869)

        model = self.create_model(input_dims=input_dims,
                            nr_layers=nr_layers,
                            nr_units=nr_units,
                            activation='elu',
                            kerner_initializer='he_normal',
                            dropout_layer=layers.Dropout(dropout_rate),
                            optimizer=optimizers.Adam(lr=lr))

        print(model.summary())
        
        mc = callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True)
        
        early_stopping = callbacks.EarlyStopping(patience=50)

        lr_scheduler = callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        
        hist = model.fit(x=X_train, 
                        y=y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=class_weights,
                        callbacks=[mc, lr_scheduler])
        
        return model, hist

    def train(self) -> None:        
        print("Starting Deep Learning Classifier")
        self.start_time = time.time()      

        trials = Trials()

        space = { 
            'epochs': hp.choice('epochs', [30]),  
            'batch_size': hp.quniform('batch_size', 512, 4096, 10),
            'nr_layers': hp.quniform('nr_layers', 4, 6, 1),  
            'nr_units': hp.quniform('nr_units', 300, 400, 100), 
            'activation': hp.choice('activation', ['elu']),
            'dropout_rate': hp.uniform('dropout_rate', 0, 0.3),
            'optimizer': hp.choice('optimizer', [
                {
                    'name': 'adam',
                    'lr_mult': hp.loguniform('adam_lr_rate_mult', -0.5, 1),
                }
            ])
        }


        fmin(fn=self.train_best_parameter_model, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Deep Learning Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        print("Saved Deep Learning")   

        print("Starting Optimized Deep Learning Classifier")
        
        self.start_time = time.time()     

        self.train_optimized_model(self.X_train, self.y_train_is_attack, self.X_val, self.y_val.label_is_attack.values, 'models/opt_model.h5',
            epochs=200, batch_size=4096, class_weights=self.class_weights)
        
        self.end_time = time.time()
        self.time_stats_file.write("---- Time stats for Optimized Deep Learning Classifier ----")
        self.time_stats_file.write("\n")
        self.time_stats_file.write("--- %s seconds---" % (self.end_time - self.start_time))
        self.time_stats_file.write("\n")
        self.time_stats_file.close()

        print("Saved Optimized Deep Learning") 


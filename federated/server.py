import flwr as fl
import pandas as pd
import pyarrow.feather as feather
import time

from sklearn.metrics import log_loss
from typing import Dict, Optional, Tuple, List, Dict
from numpy import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, metrics
import time

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'E:\\Mestrado\\askonas-ids')
from config.config import Config

# X_train = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier"+ '\\X_train.feather')
# y_train = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + '\\y_train.feather')
# X_val = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + '\\X_val.feather')
# y_val = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + "\\" + 'y_val.feather')
# X_test = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + '\\X_test.feather')
# y_test = feather.read_feather("E:\\Mestrado\\ml-ids\\notebooks\\06_dl_classifier" + "\\" + 'y_test.feather')

X_train = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\enhanced"+ '\\X_train.feather')
y_train = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\enhanced" + '\\y_train.feather')
X_val = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\enhanced" + '\\X_val.feather')
y_val = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\enhanced" + "\\" + 'y_val.feather')
X_test = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\enhanced" + '\\X_test.feather')
y_test = feather.read_feather("E:\\Mestrado\\askonas-ids\\federated\\datasets\\enhanced" + "\\" + 'y_test.feather')

input_dims = X_train.shape[1]

nr_layers = 6
nr_units = 300
dropout_rate = 0.07923141859099106
lr = (0.001 * 1.7055226678955877)
batch_size = 2610

NUM_CLIENTS = 100

def create_model(input_dims, 
                 nr_layers, 
                 nr_units, 
                 activation, 
                 kerner_initializer,
                 optimizer,
                 dropout_layer=None):
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

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        #loss, accuracy = model.evaluate(X_test, y_test.label_is_attack.values)
        e =  model.evaluate(X_test, y_test.label_is_attack.values, verbose=2, steps=5)
        e = {out: e[i] for i, out in enumerate(model.metrics_names)}
        # model.save(Config.MODELS_FOLDER + "\\federated\\10_clients\\" + '3_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '3_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '5_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '10_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '20_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '30_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '40_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '50_rounds.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '100_rounds.h5')

        model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '3_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '5_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '10_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '20_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '30_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '40_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '50_rounds_enhanced.h5')
        # model.save(Config.MODELS_FOLDER + "\\federated\\3_clients\\" + '100_rounds_enhanced.h5')
        #return loss, {"accuracy": accuracy}
        
        keys_list = list(e)
        auc_key = keys_list[1]
        precision_key = keys_list[2]
        recall_key = keys_list[3]
        loss_key = keys_list[0]
        
        #return float(e['loss']), {"auc": float(e['auc']), 'precision': float(e['precision']), 'recall': float(e['recall'])}
        return float(e[loss_key]), {"auc": float(e[auc_key]), 'precision': float(e[precision_key]), 'recall': float(e[recall_key])}

    return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":

    model_1 = create_model(input_dims=input_dims,
            nr_layers=nr_layers,
            nr_units=nr_units,
            activation='elu',
            kerner_initializer='he_normal',
            dropout_layer=layers.Dropout(dropout_rate),
            optimizer=optimizers.Adam(learning_rate=lr))

    strategy = fl.server.strategy.FedAvg(
        eval_fn=get_eval_fn(model_1),
        #fraction_fit=0.3,
        #fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3,
    )
    start = time.time()
    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:5040", config={"num_rounds": 3}, strategy=strategy)
    #fl.server.start_server("localhost:5040", config={"num_rounds": 3})
    end = time.time()
    print(end-start)


'''
def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: CatBoostClassifier):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    #_, (X_test, y_test) = utils.load_mnist()

    X_test = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\X_test.feather')
    y_test = pd.DataFrame = feather.read_feather('E:\\Mestrado\\askonas-ids\\federated\\y_test.feather')

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        # utils.set_model_params(model, parameters)
        model.get_feature_importance()
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = CatBoostClassifier()
    # utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 3})
    '''
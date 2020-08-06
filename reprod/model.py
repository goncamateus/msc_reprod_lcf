import os

import tensorflow as tf
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import GRU, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam


def create_models(input_shape: tuple, learning_rate: float,
                  models: list, horizons=12):
    """
    Creates our DNN Models.

    Parameters
    ----------
    input_shape : tuple

    lr : float
        Learning rate

    models : list
        Which models you want to train

    horizons : int
        Number of horizons you want to predict

    Returns
    -------
    list
        DNN Models

    """
    regressors = list()
    if 'MLP' in models:
        regressor = Sequential()
        regressor.add(Flatten(input_shape=input_shape))
        regressor.add(Dense(units=160, activation='relu'))
        regressor.add(Dense(units=180, activation='relu'))
        regressor.add(Dense(units=160, activation='relu'))
        regressor.add(Dense(units=240, activation='relu'))
        regressor.add(Dense(units=horizons))
        regressor.summary()
        Adam(learning_rate=learning_rate)
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressors.append(regressor)

    if 'GRU' in models:
        regressor = Sequential()
        regressor.add(GRU(units=160, activation='relu',
                          return_sequences=True, input_shape=input_shape))
        regressor.add(GRU(units=180, activation='relu', return_sequences=True))
        regressor.add(GRU(units=160, activation='relu', return_sequences=True))
        regressor.add(GRU(units=240, activation='relu'))
        regressor.add(Dense(units=horizons))
        regressor.summary()
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressors.append(regressor)

    if 'LSTM' in models:
        regressor = Sequential()
        regressor.add(LSTM(units=160, activation='relu',
                           return_sequences=True, input_shape=input_shape))
        regressor.add(LSTM(units=180, activation='relu',
                           return_sequences=True))
        regressor.add(LSTM(units=160, activation='relu',
                           return_sequences=True))
        regressor.add(LSTM(units=240, activation='relu'))
        regressor.add(Dense(units=horizons))
        regressor.summary()
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressors.append(regressor)

    return regressors


def get_callbacks(central, model, decomp_method):
    cbs = list()
    checkpoint_path = f"data/models/{central}/{decomp_method}/{model}.ckpt"
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            period=2)
    cbs.append(cp_callback)
    es_callback = callbacks.EarlyStopping(monitor='loss')
    cbs.append(es_callback)
    return cbs


def load_models(central, regressors, models, decomp_method):
    models_path = f"data/models/{central}/{decomp_method}"
    for i, model in enumerate(models):
        path = models_path + model + '.ckpt'
        regressors[i].load_weights(path)

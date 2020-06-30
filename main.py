import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from reprod.model import create_models, get_callbacks, load_models
from reprod.preprocess import prepare_data
from reprod.utils import plot_taylor


def main(dataset, test_only=False, dec=True, regvars=60, models=['MLP']):

    # Load data from .mat file and create necessary folders
    matfile = scipy.io.loadmat(dataset)
    central = dataset.split('/')[2].split('.')[0]
    if not os.path.exists(f'data/components/'):
        os.system(f'mkdir data/components/')
    if not os.path.exists(f'data/models/'):
        os.system(f'mkdir data/models/')
    if not os.path.exists(f'data/out/'):
        os.system(f'mkdir data/out/')
    if not os.path.exists(f'data/components/{central}'):
        os.system(f'mkdir data/components/{central}')
    if not os.path.exists(f'data/models/{central}'):
        os.system(f'mkdir data/models/{central}')

    # Prepare Data
    serie_nan = np.array(matfile['P'], dtype=np.float32)
    serie = serie_nan[~np.isnan(serie_nan)]
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        serie, dec, central, regvars)

    # Prepare Model and fit or load weights
    regressors = create_models(input_shape=(
        X_train.shape[1], X_train.shape[2]), learning_rate=1e-3)
    if not test_only:
        for regressor in regressors:
            regressor.fit(X_train, y_train, epochs=10,
                          batch_size=32, callbacks=get_callbacks(central))
    else:
        regressors = load_models(central)

    # Predict from data
    preds = list()
    for i, model in models:
        y_pred = regressors[i].predict(X_test)
        preds.append((model, y_pred))

    # Plot Real vs Predicted
    plt.figure()
    plt.plot(y_test/scaler.scale_)
    for _, y_pred in preds:
        plt.plot(y_pred/scaler.scale_)
    plt.legend(['Real'] + [model for model, _ in preds])
    plt.savefig(f'data/out/result_{central}.png')

    plt.clf()
    plt.figure()
    plt.plot(y_test[:1000]/scaler.scale_)
    for _, y_pred in preds:
        plt.plot(y_pred[:1000]/scaler.scale_)
    plt.legend(['Real'] + [model for model, _ in preds])
    plt.savefig(f'data/out/result_zoomed_{central}.png')
    plt.clf()

    # Plot diff between Predicted and Real
    for model, y_pred in preds:
        plt.figure()
        plt.plot(y_pred.squeeze()/scaler.scale_ - y_test/scaler.scale_)
        plt.savefig(f'data/out/diff_{central}_{model}.png')
        plt.clf()

    # Plot Taylor Diagram
    predictions_dict = {model: y_pred.squeeze(
    )/scaler.scale_ for model, y_pred in preds}
    plot_taylor(y_test/scaler.scale_, predictions_dict, central)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description='Predicts your time series')
    PARSER.add_argument('--dataset', metavar='dataset',
                        type=str, default='data/G1.mat',
                        help='Which dataset (.mat)\
                             you wanna choose from ./data')
    PARSER.add_argument('--test', metavar='test',
                        type=int, default=0,
                        help='Rather you wanna train or not your model')
    PARSER.add_argument('--decompose', metavar='dec',
                        type=int, default=1,
                        help='Rather you wanna decompose or not your serie')
    PARSER.add_argument('--regvars', metavar='regvars',
                        type=int, default=60,
                        help='How many regvars you want for your model(s)')
    PARSER.add_argument('--models', metavar='models',
                        type=str, default='MLP',
                        help='Which DNN models you wanna try (MLP, GRU, LSTM)')
    ARGS = PARSER.parse_args()
    MODELS = ARGS.models.upper().split(',')
    main(ARGS.dataset, test_only=bool(ARGS.test),
         dec=bool(ARGS.decompose), regvars=ARGS.regvars, models=MODELS)

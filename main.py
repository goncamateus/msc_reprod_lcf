import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from reprod.model import create_models, get_callbacks, load_models
from reprod.preprocess import prepare_data
from reprod.utils import plot_taylor


def main(dataset, test_only=False, dec=True,
         regvars=60, horizons=12, models=['MLP'],
         epochs=10, decomp_method='fft'):

    # Load data from .mat file and create necessary folders
    matfile = scipy.io.loadmat(dataset)
    central = dataset.split('/')[2].split('.')[0]
    if not os.path.exists(f'data/components/'):
        os.system(f'mkdir data/components/')
    if not os.path.exists(f'data/models/'):
        os.system(f'mkdir data/models/')
    if not os.path.exists(f'data/out/'):
        os.system(f'mkdir data/out/')
    if not os.path.exists(f'data/out/{central}'):
        os.system(f'mkdir data/out/{central}')
    path_out = f'data/out/{central}/{decomp_method}'
    if not os.path.exists(path_out):
        os.system(f'mkdir {path_out}')
    if not os.path.exists(path_out+'/pure'):
        os.system(f'mkdir {path_out}/pure')
    if not os.path.exists(path_out+'/zoomed'):
        os.system(f'mkdir {path_out}/zoomed')
    if not os.path.exists(path_out+'/diff'):
        os.system(f'mkdir {path_out}/diff')
    if not os.path.exists(path_out+'/taylor'):
        os.system(f'mkdir {path_out}/taylor')
    if not os.path.exists(f'data/components/{central}'):
        os.system(f'mkdir data/components/{central}')
    if not os.path.exists(f'data/models/{central}'):
        os.system(f'mkdir data/models/{central}')
    if not os.path.exists(f'data/models/{central}/{decomp_method}'):
        os.system(f'mkdir data/models/{central}/{decomp_method}')

    # Prepare Data
    serie_nan = np.array(matfile['P'], dtype=np.float32)
    serie = serie_nan[~np.isnan(serie_nan)]
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        serie, dec, central, regvars, horizons, decomp_method)

    # Prepare Model and fit or load weights
    regressors = create_models(input_shape=(X_train.shape[1],
                                            X_train.shape[2]),
                               learning_rate=1e-3, models=models,
                               horizons=horizons)
    if not test_only:
        for i, regressor in enumerate(regressors):
            regressor.fit(X_train, y_train, epochs=epochs, batch_size=128,
                          callbacks=get_callbacks(central, models[i],
                                                  decomp_method))
    else:
        load_models(central, regressors, models, decomp_method)

    # Predict from data
    preds = list()
    for i, model in enumerate(models):
        y_pred = regressors[i].predict(X_test)
        preds.append((model, y_pred))

    # Plot Real vs Predicted
    plt.figure()
    for i in range(horizons):
        real = y_test.transpose()[i]
        plt.plot(real/scaler.scale_)
        for _, y_pred in preds:
            y_pred = y_pred.transpose()[i]
            plt.plot(y_pred/scaler.scale_)
        plt.legend(['Real'] + [model for model, _ in preds])
        plt.title(f'{decomp_method.upper()} Comparisson for {i+1} horizons')
        plt.savefig(f'{path_out}/pure/horizon_{i+1}.png')
        plt.clf()

    for i in range(horizons):
        real = y_test.transpose()[i][:1000]
        plt.plot(real/scaler.scale_)
        for _, y_pred in preds:
            y_pred = y_pred.transpose()[i][:1000]
            plt.plot(y_pred/scaler.scale_)
        plt.legend(['Real'] + [model for model, _ in preds])
        plt.title(
            f'{decomp_method.upper()} Comparisson for {i+1} horizons Zoomed')
        plt.savefig(f'{path_out}/zoomed/horizon_{i+1}.png')
        plt.clf()

    # Plot diff between Predicted and Real
    for i in range(horizons):
        real = y_test.transpose()[i]
        for model, y_pred in preds:
            y_pred = y_pred.transpose()[i]
            plt.plot(y_pred/scaler.scale_ - real/scaler.scale_)
            plt.savefig(f'{path_out}/diff/{model}_horizon_{i+1}.png')
            plt.clf()

    # Plot Taylor Diagram
    for i in range(horizons):
        predictions_dict = {}
        for model, y_pred in preds:
            model = model + '_' + decomp_method.upper()
            predictions_dict[model +
                             f'_{i+1}'] = y_pred.transpose()[i]/scaler.scale_
        plot_taylor(y_test.transpose()[i]/scaler.scale_,
                    predictions_dict, i+1, path_out)
        plt.clf()


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
    PARSER.add_argument('--horizons', metavar='horizons',
                        type=int, default=12,
                        help='How many horizons you want to predict')
    PARSER.add_argument('--models', metavar='models',
                        type=str, default='MLP',
                        help='Which DNN models you wanna try (MLP, GRU, LSTM)')
    PARSER.add_argument('--epochs', metavar='epochs',
                        type=int, default=10,
                        help='Number of epochs you want to train each model')
    PARSER.add_argument('--decomp_method', metavar='decomp_method',
                        type=str, default='fft',
                        help='Method to decompose the serie')
    ARGS = PARSER.parse_args()
    MODELS = ARGS.models.upper().split(',')

    main(ARGS.dataset, test_only=bool(ARGS.test),
         dec=bool(ARGS.decompose), regvars=ARGS.regvars,
         horizons=ARGS.horizons, models=MODELS,
         epochs=ARGS.epochs, decomp_method=ARGS.decomp_method)

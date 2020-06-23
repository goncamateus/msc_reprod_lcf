import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler

from decomposition import decomp, get_periods
from model import create_model, get_callbacks, load_model
from utils import plot_taylor


def prepare_data(serie: np.ndarray, dec: bool, central: str) -> tuple:
    """
    Preprocess data and separates it in train and test.
    All the data is normalized.

    Parameters
    ----------
    serie : np.ndarray
        Time Serie data

    dec : bool
        Rather you wanna decompose or not your serie

    central : str
        The dataset name

    Returns
    -------
    tuple
        X_train, y_train, X_test, y_test, normalizer scaler
    """
    scaler = MinMaxScaler()
    comp_data = scaler.fit_transform(serie.reshape(-1, 1))
    serie = comp_data.squeeze()
    if dec:
        periods = get_periods(serie)
        print(f'Decomp serie in {periods.size} components')
        main_components = decomp(serie, periods)
        sub_comps = []
        menor = 10e20
        index = 0
        for i, comp in enumerate(main_components.transpose()[:-1]):
            periods = get_periods(comp)
            print(f'Decomp comp {i+1} in {periods.size} components')
            comps = decomp(comp, periods)
            if comps.shape[0] < menor:
                menor = comps.shape[0]
                index = i
            sub_comps.append(comps)
            print(f'saving comp {i}')
            with open('data/components/{}/sub_comps_{}.pkl'.format(
                    central, i), 'wb') as pf:
                pickle.dump(comps, pf)
    else:
        sub_comps = []
        menor = 10e20
        index = 0
        for i, fp in enumerate(os.listdir('data/components/{}/'.format(central))):
            with open('data/components/{}/'.format(central)+fp, 'rb') as pf:
                comps = pickle.load(pf)
                sub_comps.append(comps)
                if comps.shape[0] < menor:
                    menor = comps.shape[0]
                    index = i
    data = np.zeros((sub_comps[index].shape[0], 1))
    for i in range(len(sub_comps)):
        if i == 0:
            data = np.concatenate((data, sub_comps[i][:menor]), 1)
            data = data[:, 1:]
        else:
            data = np.concatenate((data, sub_comps[i][:menor]), 1)

    X = []
    y = serie[60:data.shape[0]]

    for i in range(60, data.shape[0]):
        X.append(data[i-60:i])

    X, y = np.array(X), np.array(y)
    X_train = X[:int(len(X)*0.7)]
    X_test = X[int(len(X)*0.7):]
    y_train = y[:int(len(y)*0.7)]
    y_test = y[int(len(y)*0.7):]
    return X_train, y_train, X_test, y_test, scaler


def main(dataset, test_only=False, dec=True):

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
        serie, dec, central)

    # Prepare Model and fit or load weights
    regressor = create_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                             lr=1e-4)
    if not test_only:
        regressor.fit(X_train, y_train, epochs=5,
                      batch_size=128, callbacks=get_callbacks(central))
    else:
        load_model(regressor, central)

    # Predict from data
    y_pred = regressor.predict(X_test)

    # Plot Real vs Predicted
    plt.figure()
    plt.plot(y_test/scaler.scale_)
    plt.plot(y_pred/scaler.scale_)
    plt.legend(['Real', 'MLP'])
    plt.savefig(f'data/out/result_{central}.png')
    plt.clf()

    # Plot diff between Predicted and Real
    plt.figure()
    plt.plot(y_pred.squeeze()/scaler.scale_ - y_test/scaler.scale_)
    plt.savefig(f'data/out/diff_{central}.png')
    plt.clf()

    # Plot Taylor Diagram
    predictions_dict = {'MLP': y_pred.squeeze()/scaler.scale_}
    plot_taylor(y_test/scaler.scale_, predictions_dict, central)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predicts your time series')
    parser.add_argument('--dataset', metavar='dataset', type=str, default='data/G1.mat',
                        help='Which dataset (.mat) you wanna choose from ./data')
    parser.add_argument('--test', metavar='test', type=int,
                        default=0, help='Rather you wanna train or not your model')
    parser.add_argument('--decompose', metavar='dec', type=int, default=1,
                        help='Rather you wanna decompose or not your serie')
    args = parser.parse_args()
    main(args.dataset, test_only=bool(args.test), dec=bool(args.decompose))

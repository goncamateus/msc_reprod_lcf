import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from decomposition import decomp, get_periods


def get_comps(serie, sub=False):
    periods = get_periods(serie)
    plus_str = 'sub comp' if sub else 'serie'
    print(f'Decomp {plus_str} in {periods.size} components')
    components = decomp(serie, periods)

    return components


def get_sub_comps(main_components, central):
    sub_comps = []
    menor = 10e20
    index = 0
    for i, comp in enumerate(main_components.transpose()[:-1]):
        comps = get_comps(comp, sub=True)
        if comps.shape[0] < menor:
            menor = comps.shape[0]
            index = i
        sub_comps.append(comps)
        print(f'saving comp {i}')
        with open('data/components/{}/sub_comps_{}.pkl'.format(
                central, i), 'wb') as pf:
            pickle.dump(comps, pf)

    return sub_comps, index, menor


def load_sub_comps(central):
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
    return sub_comps, index, menor


def set_data(serie, sub_comps, index, menor):
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

    return np.array(X), np.array(y)


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
        main_components = get_comps(serie)
        sub_comps, index, menor = get_sub_comps(main_components, central)
    else:
        sub_comps, index, menor = load_sub_comps(central)

    X, y = set_data(serie, sub_comps, index, menor)
    X_train = X[:int(len(X)*0.7)]
    X_test = X[int(len(X)*0.7):]
    y_train = y[:int(len(y)*0.7)]
    y_test = y[int(len(y)*0.7):]
    return X_train, y_train, X_test, y_test, scaler

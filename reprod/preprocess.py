import os
import pickle

import numpy as np
from scipy.signal import cwt, ricker
from sklearn.preprocessing import MinMaxScaler

from reprod.decomposition import decomp, get_periods


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
    for i, fp in enumerate(os.listdir(
            'data/components/{}/'.format(central))):
        with open('data/components/{}/'.format(central)+fp, 'rb') as pf:
            comps = pickle.load(pf)
            sub_comps.append(comps)
            if comps.shape[0] < menor:
                menor = comps.shape[0]
                index = i
    return sub_comps, index, menor


def set_data(serie, sub_comps, index, menor, reg_vars=60, horizons=12):
    data = np.zeros((sub_comps[index].shape[0], 1))
    for i in range(len(sub_comps)):
        if i == 0:
            data = np.concatenate((data, sub_comps[i][:menor]), 1)
            data = data[:, 1:]
        else:
            data = np.concatenate((data, sub_comps[i][:menor]), 1)

    X = []
    y = list()
    for i in range(reg_vars, data.shape[0]-horizons+1):
        obs_y = [serie[i+j] for j in range(horizons)]
        y.append(obs_y)

    for i in range(reg_vars, data.shape[0]-horizons+1):
        X.append(data[i-reg_vars:i])

    return np.array(X), np.array(y)


def set_data_cwt(serie, wttrain, wttest, scaler, horizons, reg_vars):
    X_train = list()
    X_test = list()
    y = list()
    for i in range(reg_vars, wttrain.shape[0]-horizons+1):
        obs_y = [serie[i+j] for j in range(horizons)]
        y.append(obs_y)
    y_train = np.array(y)
    y_train = scaler.transform(y_train)

    y = list()
    for i in range(reg_vars, wttest.shape[0]-horizons+1):
        obs_y = [serie[wttrain.shape[0]+i+j] for j in range(horizons)]
        y.append(obs_y)
    y_test = np.array(y)
    y_test = scaler.transform(y_test)

    for i in range(reg_vars, wttrain.shape[0]-horizons+1):
        X_train.append(wttrain[i-reg_vars:i])

    for i in range(reg_vars, wttest.shape[0]-horizons+1):
        X_test.append(wttest[i-reg_vars:i])

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_train, y_train, X_test, y_test


def prepare_data(serie: np.ndarray, dec: bool,
                 central: str, reg_vars: int,
                 horizons: int, decomp_method: str) -> tuple:
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

    reg_vars : int
        How many regvars you want for your model(s)

    horizons : int
        Number of horizons you want to predict

    decomp_method : str
        Lucas method or CWT method

    Returns
    -------
    tuple
        X_train, y_train, X_test, y_test, normalizer scaler
    """
    if decomp_method == 'fft':
        scaler = MinMaxScaler()
        comp_data = scaler.fit_transform(serie.reshape(-1, 1))
        serie = comp_data.squeeze()
        if dec:
            main_components = get_comps(serie)
            sub_comps, index, menor = get_sub_comps(main_components, central)

        else:
            sub_comps, index, menor = load_sub_comps(central)

        X, y = set_data(serie, sub_comps, index, menor,
                        reg_vars=reg_vars, horizons=horizons)
        X_train = X[:int(len(X)*0.7)]
        X_test = X[int(len(X)*0.7):]
        y_train = y[:int(len(y)*0.7)]
        y_test = y[int(len(y)*0.7):]
    else:
        train_data = serie[:int(len(serie)*2/3)]
        test_data = serie[int(len(serie)*2/3):]
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data.reshape(-1, 1)).squeeze()
        test_data = scaler.transform(test_data.reshape(-1, 1)).squeeze()
        widths = np.arange(1, 65)
        wttrain = cwt(train_data, ricker, widths).transpose()
        wttest = cwt(test_data, ricker, widths).transpose()
        X_train, y_train, X_test, y_test = set_data_cwt(serie, wttrain,
                                                        wttest, scaler,
                                                        horizons, reg_vars)
    return X_train, y_train, X_test, y_test, scaler

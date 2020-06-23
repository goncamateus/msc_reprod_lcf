import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import skill_metrics as sm
import tensorflow as tf
from scipy.signal import find_peaks, hilbert
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


def get_periods(serie: np.ndarray) -> np.ndarray:
    """
    Extract periods with most importance in 
    the Fourier Transformed Serie.

    Parameters
    ----------
    serie : np.ndarray
        The serie which you will analyze.

    Returns
    -------
    np.ndarray
        Periods with most importance in FFTed serie.

    """
    fft_serie = np.fft.fft(serie)
    fft_serie = fft_serie[:int(fft_serie.size/2)]
    fft_serie = hilbert(np.abs(fft_serie))
    # ts is set to 30 days here
    # set it for your purpose
    ts = 30/(24*60)
    peaks_idx = find_peaks(fft_serie)[0]
    peaks_idx = np.array(peaks_idx)
    periodos = np.round(1/(peaks_idx*ts))
    periodos = sorted(periodos, reverse=True)
    r_per = []
    for per in periodos:
        if per not in r_per and per > 1:
            r_per.append(per)
    periodos = np.array(r_per, dtype=np.int64)
    return periodos


def decomp(serie: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Decompose the serie according to the given periods.


    Parameters
    ----------
    serie : np.ndarray
        The serie which you will decompose.

    periods : np.ndarray
        Periods of reference

    Returns
    -------
    np.ndarray
        Decomposed Serie in #periods components

    """
    o = np.zeros((periods[0], periods.size + 1))
    comp_dados = np.zeros((serie.size - periods[0], periods.size + 1))
    for i in range(serie.size - periods[0]):
        o[:, 0] = serie[i:periods[0]+i]
        for j in range(periods.size):
            comp_dados[i, j] = np.mean(o[periods[0] - periods[j]:, j])
            o[:, j+1] = o[:, j] - comp_dados[i, j]
    return comp_dados


def create_model(input_shape: tuple, lr: float):
    """
    Creates our DNN Model.

    Parameters
    ----------
    input_shape : tuple

    lr : float
        Learning rate

    Returns
    -------
    MLP model

    """
    regressor = Sequential()
    regressor.add(Flatten(input_shape=input_shape))
    regressor.add(Dense(units=160, activation='relu'))
    regressor.add(Dense(units=180, activation='relu'))
    regressor.add(Dense(units=160, activation='relu'))
    regressor.add(Dense(units=240, activation='relu'))
    regressor.add(Dense(units=1))
    regressor.summary()
    Adam(learning_rate=lr)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor


def plot_taylor(ref: np.ndarray, predictions_dict: dict, central: str):
    """
    Plots Taylor Diagram refering to Reference Serie
    Code by: hrc and jvsg (@cin.ufpe.br)

    Parameters
    ----------
    ref : np.ndarray
        Reference Serie to Taylor Diagram

    predictions_dict : dict
        Dictionary with the predictions you made.
        e.g: {'MLP': np.array([1,2,3,4,5]), 'LSTM': np.array([1,2,3,4,5])}

    Returns
    -------
    None

    """
    data = {'preds': [v for k, v in predictions_dict.items()],
            'ref': ref}

    taylor_stats = []
    for pred in data['preds']:
        taylor_stats.append(sm.taylor_statistics(pred, data['ref'], 'data'))

    sdev = np.array([taylor_stats[0]['sdev'][0]]+[x['sdev'][1]
                                                  for x in taylor_stats])
    crmsd = np.array([taylor_stats[0]['crmsd'][0]]+[x['crmsd'][1]
                                                    for x in taylor_stats])
    ccoef = np.array([taylor_stats[0]['ccoef'][0]]+[x['ccoef'][1]
                                                    for x in taylor_stats])

    # To change other params in the plot, check SkillMetrics documentation in
    # https://github.com/PeterRochford/SkillMetrics/wiki/Target-Diagram-Options
    sm.taylor_diagram(sdev, crmsd, ccoef, styleOBS='-',
                      colOBS='g', markerobs='o',
                      titleOBS='Observation', markerLabel=['placeholder']+[k for k, v in predictions_dict.items()])
    plt.savefig(f'out/taylor_{central}.png')


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
    central = dataset.split('/')[1].split('.')[0]
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
    checkpoint_path = f"data/models/{central}/model_{central}.ckpt"
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=False,
                                            period=2)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss')
    if not test_only:
        regressor.fit(X_train, y_train, epochs=5,
                      batch_size=128, callbacks=[cp_callback, es_callback])
    else:
        latest = tf.train.latest_checkpoint(f"data/models/{central}")
        regressor.load_weights(latest)

    # Predict from data
    y_pred = regressor.predict(X_test)

    # Plot Real vs Predicted
    plt.figure()
    plt.plot(y_test/scaler.scale_)
    plt.plot(y_pred/scaler.scale_)
    plt.legend(['Real', 'MLP'])
    plt.savefig(f'out/result_{central}.png')
    plt.clf()

    # Plot diff between Predicted and Real
    plt.figure()
    plt.plot(y_pred.squeeze()/scaler.scale_ - y_test/scaler.scale_)
    plt.savefig(f'out/diff_{central}.png')
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

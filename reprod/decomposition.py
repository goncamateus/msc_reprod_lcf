import numpy as np
from scipy.signal import find_peaks, hilbert


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

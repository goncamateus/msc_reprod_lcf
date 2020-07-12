import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm


def plot_taylor(ref: np.ndarray, predictions_dict: dict,
                central: str, horizon: int):
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

    central : str
        The name of the dataset

    horizon : int
        The index of horizon

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
                      titleOBS='Observation',
                      markerLabel=['placeholder']+[k for k, v in predictions_dict.items()])
    plt.savefig(f'data/out/taylor_{central}_horizon_{horizon}.png')

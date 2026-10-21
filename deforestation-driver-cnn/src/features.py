import numpy as np


def handcrafted(X):
    """Fitur untuk baseline ML konvensional: mean/std/min/max per channel + luas area terganggu."""
    f = [X.mean((2, 3)), X.std((2, 3)), X.min((2, 3)), X.max((2, 3))]
    vv = X[:, 4]
    area = (vv < (np.median(vv.reshape(len(X), -1), axis=1)[:, None, None] - 0.75)).mean((1, 2))
    return np.concatenate(f + [area[:, None]], axis=1)

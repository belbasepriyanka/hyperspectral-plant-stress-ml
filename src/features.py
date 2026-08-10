import numpy as np
import pandas as pd

def continuum_normalize(X: np.ndarray) -> np.ndarray:
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    return (X - mins) / (maxs - mins + 1e-9)

def spectral_features(wavelengths, X):
    w = np.asarray(wavelengths)
    def nearest(target):
        return int(np.argmin(np.abs(w-target)))
    r550, r670, r705, r740, r800 = [X[:, nearest(v)] for v in (550,670,705,740,800)]
    return pd.DataFrame({
        "R550": r550,
        "R670": r670,
        "R705": r705,
        "R740": r740,
        "R800": r800,
        "NDVI_like": (r800-r670)/(r800+r670+1e-9),
        "NDRE705": (r800-r705)/(r800+r705+1e-9),
        "red_edge_slope": (r740-r705)/(740-705)
    })

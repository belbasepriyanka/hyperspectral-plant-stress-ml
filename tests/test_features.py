import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from features import continuum_normalize

def test_normalization():
    X = np.array([[2.,4.,6.]])
    out = continuum_normalize(X)
    assert np.isclose(out.min(), 0)
    assert np.isclose(out.max(), 1)

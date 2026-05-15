import numpy as np

def one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=int)
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if np.any(y < 0):
        raise ValueError("Labels must be non-negative integers")
    
    K = int(np.max(y)) + 1 if num_classes is None else int(num_classes)
    
    if K < 1:
        raise ValueError(f"num_classes must be >= 1, got {K}")
    if np.any(y >= K):
        raise ValueError(f"Label {np.max(y)} >= num_classes={K}")
    
    N = y.shape[0]
    out = np.zeros((N, K), dtype=float)
    out[np.arange(N), y] = 1.0
    return out
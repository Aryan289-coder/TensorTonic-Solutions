import numpy as np

def euclidean_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape")
    
    return float(np.sqrt(np.sum((a - b) ** 2)))
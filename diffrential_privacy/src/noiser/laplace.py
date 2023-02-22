import pandas as pd
import numpy as np
# from scipy import stats

def laplace_noise(epsilon,sensitivity,size=None):
    
    scale = sensitivity / epsilon
    if size:
        return np.random.laplace(loc=0, scale=scale,size=size)
    else:
        return np.random.laplace(loc=0, scale=scale)



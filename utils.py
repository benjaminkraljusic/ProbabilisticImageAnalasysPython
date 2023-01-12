# Useful functions
import numpy as np

def gaussNormalDensityFcn(x, mu, sigma2):
    y = 1/(np.sqrt(2*np.pi*sigma2))*np.exp(-1/2 * (x - mu)**2/sigma2)
    return y
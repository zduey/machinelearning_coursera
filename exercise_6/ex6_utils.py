# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plotData(X, y):
    """
    Plots the data points X and y into a new figure with + for the positive
    examples and 0 for the negative examples. X is assumed to be an Mx2 array.

    """
    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]

    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")

    return (fig, ax)

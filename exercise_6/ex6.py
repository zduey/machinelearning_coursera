# Machine Learning Coursera -- Exercise 6

## Initialization
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex5_utils import *
import scipy.io

# Part 1 -- Loading and Visualizing Data
raw_mat = scipy.io.loadmat("ex6data1.mat")
X = raw_mat.get("X")
y = raw_mat.get("y").flatten()

fig, ax = plotData(X,y)
fig.show()

# Part 2 -- Training Linear SVM
c = 1
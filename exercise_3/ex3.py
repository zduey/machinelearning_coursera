# Machine Learning Coursera -- Exercise 3, Part 1

## Initialization
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex3_utils import *
import scipy.io


# Set up parameters
input_layer_size = 400
num_labels = 10


# Load training data
print("Loading training data...")

raw_mat = scipy.io.loadmat("ex3data1.mat")
X = raw_mat.get("X")
y = raw_mat.get("y").flatten()
m = np.hstack((np.ones((len(y),1)),X))# add column of ones

# Randomly select 100 datapoints to display
rand_indices = np.random.randint(0,len(m),100)
sel = X[rand_indices,:] 


# Display the data
digit_grid, ax = displayData(sel)
digit_grid.show()

input("Program paused, press enter to continue...")


# ============ Part 2: Vectorize Logistic Regression ============
reg_param = 1.0
theta = oneVsAll(m,y,10,reg_param)


# Machine Learning Coursera -- Exercise 4

## Initialization
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex4_utils import *
import scipy.io


# Set up parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10


# Load training data
print("Loading training data...")

raw_mat = scipy.io.loadmat("ex4data1.mat")
X = raw_mat.get("X")
y = raw_mat.get("y").flatten()
y = (y - 1) % 10 # hack way of fixing conversion from 1-indexing


# Randomly select 100 datapoints to display
rand_indices = np.random.randint(0,len(X),100)
sel = X[rand_indices,:] 


# Display the data
digit_grid, ax = displayData(sel)
digit_grid.show()

input("Program paused, press enter to continue...")
print("Loading neural network parameters \n")

raw_params = scipy.io.loadmat("ex4weights.mat")
theta1 = raw_params.get("Theta1") # 25 x 401
theta2 = raw_params.get("Theta2") # 10 x 26

# Unroll Parameters
nn_params = np.append(theta1,theta2).reshape(-1)

# Compute Cost -- Feed Forward
reg_param = 1.0
cost = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
		   X,y,reg_param)

print("The initial cost is: ", cost)

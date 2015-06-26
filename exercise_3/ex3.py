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
y[y== 10] = 0

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

predictions = predictOneVsAllAccuracy(theta,m)
accuracy = np.mean(y == predictions) * 100
print("Training Accuracy with logit: ", accuracy, "%")
input("Program pauses, press enter to continue...")

# =================== Part 3: Neural Networks ===================

# Load pre-estimated weights
print("Loading saved neural networks parameters...")
raw_params = scipy.io.loadmat("ex3weights.mat")
theta1 = raw_params.get("Theta1") # 25 x 401
theta2 = raw_params.get("Theta2") # 10 x 26


# Note: Parameters in theta1,theta2 are based on 1 indexing. To make it work,
#       we need to either adjust theta1 and theta2, or manipulate the
#       predictions. Solution is to add 1 and take the mod with resepct to
#       10 so 10s become zeros and everything else gets bumped up one.

predictions = (predict(theta1,theta2,X) + 1) % 10
accuracy = np.mean(y == predictions) * 100
print("Training Accuracy with neural network: ", accuracy, "%")



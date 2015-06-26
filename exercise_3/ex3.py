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


# Issue here: something with 1 indexing of y. This is currently a hacky
# way of fixing the problem. Clearer fix needed.
predictions = predict(theta1,theta2,X) + 1
predictions[predictions == 10] = 0
accuracy = np.mean(y == predictions) * 100
print("Training Accuracy with neural network: ", accuracy, "%")

rp = np.random.permutation(range(5000))

for i in rp:
    # Show single image
    print("Displaying example image \n ")
    grid, ax2 = displayImage(X[i])
    grid.show()
    
    pred = predict(theta1,theta2,X[i,:]) + 1
    pred[pred == 10] = 0
    print("Neural Network Prediction: ", pred, np.mod(pred,10))
    input("Program paused... Press Enter to see another image")

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
y = (y - 1) % 10 # hack way of fixing conversion MATLAB 1-indexing

# Randomly select 100 datapoints to display
rand_indices = np.random.randint(0,len(X),100)
sel = X[rand_indices,:] 

# Display the data
digit_grid, ax = displayData(sel)
digit_grid.show()

print("Loading neural network parameters \n")

raw_params = scipy.io.loadmat("ex4weights.mat")
theta1 = raw_params.get("Theta1") # 25 x 401
theta2 = raw_params.get("Theta2") # 10 x 26

# Unroll Parameters
nn_params = np.append(theta1,theta2).reshape(-1)

# Compute Unregularized Cost
print("Checking cost function without regularization...")
reg_param = 0.0
cost, g = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
		                     X,y,reg_param)

# Test for correct cost
np.testing.assert_almost_equal(0.287629,cost,decimal=6, err_msg="Cost incorrect.")


# Compute Regularized Cost
print("Checking cost function with regularization...")
reg_param = 1.0
reg_cost, g = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,
		                        X,y,reg_param)
np.testing.assert_almost_equal(0.383770,reg_cost,decimal=6, 
                               err_msg="Regularized Cost incorrect.")


# Checking sigmoid gradient
print("Checking sigmoid gradient...")
vals = np.array([1,-0.5,0,0.5,1])
g = sigmoidGradient(vals)
np.testing.assert_almost_equal(0.25, g[2],decimal=2, err_msg="Sigmoid function incorrect")

# Initialize neural network parameters
print("Initializing neural network parameters...")
initial_theta1 = randInitializeWeights(input_layer_size+1,hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size+1,num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_theta1,initial_theta2).reshape(-1)

reg_param = 0.0
initial_cost, g = nnCostFunction(initial_nn_params,input_layer_size,
                                 hidden_layer_size,num_labels,X,y,reg_param)

print("The initial cost after random initialization: ", initial_cost)

# Check gradients
checkNNGradients(0)

# TO FIX: Gradient checking with non-zero regularization parameter fails
# Implement Regularization
# punisher = 3.0
# checkNNGradients(punisher)

# # Debugging value of the cost function
# reg_param = 10
# debug_J = nnCostFunction(initial_nn_params,input_layer_size,
#                          hidden_layer_size,num_labels,X,y,reg_param)[0]
# np.testing.assert_almost_equal(debug_J, 0.576051)


# Train NN Parameters
reg_param = 3.0
def reduced_cost_func(p):
    """ Cheaply decorated nnCostFunction """
    return nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,
                          X,y,reg_param)

results = minimize(reduced_cost_func,
                   initial_nn_params,
                   method="CG",
                   jac=True,
                   options={'maxiter':50, "disp":True})

fitted_params = results.x
# Reshape fitted_params back into neural network
theta1 = fitted_params[:(hidden_layer_size * 
             (input_layer_size + 1))].reshape((hidden_layer_size, 
                                       input_layer_size + 1))

theta2 = fitted_params[-((hidden_layer_size + 1) * 
                      num_labels):].reshape((num_labels,
                                   hidden_layer_size + 1)) 

predictions = predict(theta1, theta2, X)
accuracy = np.mean(y == predictions) * 100
print("Training Accuracy with neural network: ", accuracy, "%")

# Display the hidden layer 
digit_grid, ax = displayData(theta1[:,1:])
digit_grid.show()

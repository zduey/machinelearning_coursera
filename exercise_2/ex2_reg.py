# Machine Learning Coursera -- Exercise 2 Logistic Regression


# Initialization
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex2_utils import *


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('ex2data2.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])

## Plot Data
fig, ax = plotData(X, y)

# Specified in plot order
ax.legend(['Pass', 'Fail'])

# Labels
ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')

fig.show()

## Part 1 -- Regularized Logistic Regression
X = mapFeatureVector(X[:,0],X[:,1])

# Initialize fitting parameters
initial_theta = np.zeros(len(X[0,:]))

# Set regularization parameter to 1
reg_param = 1.0

# Optimize for theta
res = minimize(costFunctionReg,
	       initial_theta,
	       method='Newton-CG',
	       args=(X,y,reg_param),
	       jac=True, 
	       options={'maxiter':400,
			'disp':True})

theta = res.x



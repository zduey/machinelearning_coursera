# Extra functions for use in ex3.py

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def displayData(X):
    """
    Displays 2D data stored in design matrix in a nice grid.
    """
    fig, ax = plt.subplots(10,10,sharex=True,sharey=True)
    img_num = 0
    for i in range(10):
        for j in range(10):
            # Convert column vector into 20x20 pixel matrix
            # You have to transpose to have them display correctly
            img = X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            img_num += 1

    return (fig, ax)

def sigmoid(z):
    """ Returns the value from using x in the sigmoid function """
    
    return 1.0/(1 +  np.e**(-z))

def lrCostFunction(theta,X,y,reg_param):
    """
    Computes loss using sum of square errors for logistic regression
    using theta as the parameter vector for linear regression to fit 
    the data points in X and y with penalty reg_param.
    """

    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m +
       (reg_param/m)*np.sum(theta**2))

    # Don't bother with the gradient. Let scipy compute numerical
    # derivatives for you instead

    return J

def oneVsAll(X, y, num_labels, reg_param):
    """"
    Calculates parameters for num_labels individual regularized logistic 
    regression classifiers using training data X and labels y.
    """
    n = np.size(X,1)
    theta = np.zeros((n,num_labels))

    for c in range(1,num_labels+1):
        outcome = np.array(y == c).astype(int)
        initial_theta = theta[:,c-1]
        res = minimize(lrCostFunction,
                       initial_theta,
                       #method='Newton-CG',
                       args=(X,outcome,reg_param),
                       jac=False, 
                       options={'maxiter':400,
                                'disp':True})

        theta[:,c-1] = res.x
        
    return theta


def predictOneVsAllAccuracy(est_theta,X):
    return

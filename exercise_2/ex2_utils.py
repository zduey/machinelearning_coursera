# Extra functions for use in ex2.py

# Imports
import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):

    """
    Plots the data points X and y into a new figure with + for the positive
    examples and 0 for the negative examples. X is assumed to be an Mx2 array.

    """

    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]

    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"y+",neg[:,0],neg[:,1],"bo")

    # Labels and Legend
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')

    # Specified in plot order
    ax.legend(['Admitted', 'Not admitted'])

    return fig

"""
def costFunction(theta,X,y):
    
    Computs  using sum of square errors for linear 
    regression using theta as the parameter vector for 
    linear regression to fit the data points in X and y.

    Note: Requires numpy in order to run, but is not imported as part of this
	  script since it is imported in ex1.py and therefore numpy is part of
	  the namespace when the function is actually run.
    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)


    return
"""

def sigmoid(z):
    """ Returns the value from using x in the sigmoid function """
    
    return 1.0/(1 +  np.e**(-z))



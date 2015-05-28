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
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")

    # Labels and Legend
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')

    # Specified in plot order
    ax.legend(['Admitted', 'Not admitted'])

    return (fig, ax)

def costFunction(theta,X,y):
    """
    Computes loss using sum of square errors for logistic regression
    using theta as the parameter vector for linear regression to fit 
    the data points in X and y.

    """

    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m

    # Gradient
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0))/m
  
    return (J, grad)

def sigmoid(z):
    """ Returns the value from using x in the sigmoid function """
    
    return 1.0/(1 +  np.e**(-z))


def predict(theta,X):
    """
    Given a vector of parameter results and training set X,
    returns the model prediction for admission. If predicted
    probability of admission is greater than .5, predict will
    return a value of 1.
    """
    return np.where(np.dot(X,theta) > 5.,1,0)



def plotDecisionBoundary(theta, X, y):
    """	
    Plots the data points X and y into a new figure with the decision boundary
    defined by theta with + for the positive examples and o for the negative 
    examples. X is assumed to be a either 
    1) Mx3 matrix, where the first column is an all-ones column for the 
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """
    
    # Start with same plot as before
    fig, ax = plotData(X[:,1:], y)

    """
    if size(X, 2) <= 3
	# Only need 2 points to define a line, so choose two endpoints
	plot_x = [min(X(:,2))-2,  max(X(:,2))+2]

	# Calculate the decision boundary line
	plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1))

	# Plot, and adjust axes for better viewing
	plot(plot_x, plot_y)
	
	# Legend, specific for the exercise
	legend('Admitted', 'Not admitted', 'Decision Boundary')
	axis([30, 100, 30, 100])
    else
	# Here is the grid range
	u = linspace(-1, 1.5, 50)
	v = linspace(-1, 1.5, 50)

	z = zeros(length(u), length(v))
	# Evaluate z = theta*x over the grid
	for i = 1:length(u)
	    for j = 1:length(v)
		z(i,j) = mapFeature(u(i), v(j))*theta
	    end
	end
	z = z' # important to transpose z before calling contour

	# Plot z = 0
	# Notice you need to specify the range [0, 0]
	contour(u, v, z, [0, 0], 'LineWidth', 2)
    """
    return fig, ax

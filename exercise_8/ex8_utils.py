import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def estimateGaussian(X):
	mu = np.mean(X, axis=0, keepdims=True)
	sigma2 = np.var(X, axis=0, keepdims=True)

	return (mu, sigma2)

def multivariateGaussian(X, mu, sigma2):
	k = np.size(mu,1)
	if ((np.size(sigma2,0) == 1) | (np.size(sigma2,1) == 1)):
		sigma2 = np.diagflat(sigma2)

	# De-mean data 
	X = X - mu

	# Calculate p-values of data
	p = ((1 / (2* (np.pi)**(-k / 2) * np.linalg.det(sigma2)**(-.5))) *
		np.exp(-.5 * np.sum(np.dot(X, np.linalg.inv(sigma2)) * X, 1)))

	return p

def visualizeFit(X, mu, sigma2):
	meshvals = np.arange(0, 35, .5)
	X1, X2 = np.meshgrid(meshvals, meshvals)
	Z = np.hstack((X1.reshape((-1,1)), X2.reshape((-1,1))))
	Z = multivariateGaussian(Z, mu, sigma2).reshape(np.shape(X1))

	mylevels = np.array([10**i for i in np.arange(-20,0,3)])
	fig, ax = plt.subplots(1)
	ax.plot(X[:, 0], X[:, 1], 'bx')
	ax.contour(X1, X2, Z, mylevels)

	return fig, ax

def selectThreshold(yval, pval):
	bestEpsilon = 0
	bestF1 = 0
	F1 = 0

	stepsize = (np.max(pval) - np.min(pval)) / 1000
	evals = np.arange(np.min(pval), np.max(pval), stepsize)
	for epsilon in evals:
		predictions = (pval < epsilon).reshape((-1,1))
		X = np.hstack((predictions, yval))
		fp = np.sum((X[:,0] == 1) & (X[:,1] == 0))
		tp = np.sum((X[:,0] == 1) & (X[:,1] == 1))
		fn = np.sum((X[:,0] == 0) & (X[:,1] == 1))
		prec = tp / (tp + fp)
		rec = tp / (tp + fn)
		F1 = (2 * prec * rec) / (prec + rec)

		if F1 > bestF1:
			bestF1 = F1
			bestEpsilon = epsilon

	return (bestEpsilon, bestF1)

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, reg):
	# Unfold the U and W matrices from params
	X = params[:num_movies * num_features].reshape((num_movies, num_features))
	Theta = params[num_movies * num_features:].reshape((num_users, num_features))
	
	# Cost
	J = (.5 * np.sum(((np.dot(Theta,X.T).T - Y) * R)**2) + 
	    ((reg / 2) * np.sum(Theta**2)) +
	    ((reg / 2) * np.sum(X**2)))
	
	# Gradients
	X_grad = np.zeros_like(X)
	for i in range(num_movies):
		idx = np.where(R[i,:]==1)[0] # users who have rated movie i
		temp_theta = Theta[idx,:] # parameter vector for those users 
		temp_Y = Y[idx, :] # ratings given to movie i
		X_grad[i,:] = np.sum(np.dot(np.dot(temp_theta, X[i, :]) - temp_Y.T,
		    temp_theta) + reg*X[i,:], axis=0)

	Theta_grad = np.zeros_like(Theta)
	for j in range(num_users):
		idx = np.where(R[:,j]==1)[0]
		temp_X = X[idx,:]
		temp_Y = Y[idx,j]
		Theta_grad[j,:] = np.sum(np.dot(np.dot(Theta[j], temp_X.T) -
		    temp_Y, temp_X) + reg*Theta[j], axis=0) 
	grad = np.append(X_grad.flatten(), Theta_grad.flatten())
	
	return (J, grad)

def computeNumericalGradient(J,theta):
    """
    Computes the gradient of J around theta using finite differences and 
    yields a numerical estimate of the gradient.
    """
    
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4
    
    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
	
        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1)/(2 * tol)
        perturb[p] = 0

    return numgrad

def checkCostFunction(reg):
    # Create small problem
    X_t = np.random.random((4,3))
    Theta_t = np.random.random((5,3))

    # Zap out most entries
    Y = np.dot(Theta_t, X_t.T)
    Y[(np.random.random(np.shape(Y)) > .5)] = 0
    R = np.zeros_like(Y)
    R[Y != 0] = 1

    # Run gradient checking
    X = np.random.random(np.shape(X_t))
    Theta = np.random.random(np.shape(Theta_t))
    num_users = np.size(Y, 1)
    num_movies = np.size(Y,0)
    num_features = np.size(Theta_t,1)

    params = np.append(X.flatten(), Theta.flatten())
    
    def reducedCofiCostFunc(p):
        """ Cheaply decorated cofiCostFunction """
        return cofiCostFunc(p,Y, R, num_users, num_movies, num_features,0)[0]

    numgrad = computeNumericalGradient(reducedCofiCostFunc,params)

    J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
    
    # Check two gradients
    # TO FIX: Either gradient checking or actual gradient calculations are off
    np.testing.assert_almost_equal(grad, numgrad)

    return

def normalizeRatings(Y, R):
    m, n = np.shape(Y)
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros_like(Y)
    for i in range(m):
        idx = (R[i] == 1)
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]

    return (Ynorm, Ymean)








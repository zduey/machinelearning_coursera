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


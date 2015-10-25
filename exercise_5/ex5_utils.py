import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def linearRegCostFunction(init_theta,X,y,reg):
	m = len(y)
	# Make theta 2-d to get cost and gradient
	theta = init_theta[:,None]
	cost = ((np.sum((np.dot(X,theta) - y)**2))/(2*m) + 
			(reg/(2*m))*np.sum(theta**2))

	# Regularized gradient
	gradients = ((np.sum((np.dot(X,theta)-y)*X,axis=0)/m)+
		(reg/m)*theta.T)

	# Replace gradient for theta_0 with non-regularized gradient
	gradients[0] = gradients[0] - (reg/m)*theta.T
	gradients = gradients.flatten()

	return (cost, gradients)

def trainLinearReg(X,y,reg):
	initial_theta = np.zeros(np.size(X,1))
	
	res = minimize(linearRegCostFunction,
		initial_theta,
		args=(X,y,reg),
		jac=True,
		options={'maxiter':400,'disp':True})

	return res.x

def learningCurve(X,y,Xval,yval,reg):
	m_train = len(X)
	m_val = len(Xval)
	error_train = np.zeros(m_train)
	error_val = np.zeros(m_train)
	for i in range(1,m_train+1):
		est_theta = trainLinearReg(X[0:i],y[0:i],reg)
		error_train[i-1] = (np.sum((np.dot(X[0:i],est_theta)-y[0:i])**2))/(2.0*i)
		error_val[i-1] = (np.sum((np.dot(Xval,est_theta)-yval)**2))/(2.0*m_val)

	return (error_train, error_val)

def polyFeatures(X,p):
	X_poly = np.zeros((len(X),p))
	X = X.flatten()
	for i in range(1,p+1):
		X_poly[:,i-1] = X**i
	return X_poly

def featureNormalize(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0)
	normalized_X = np.divide(X - mu,sigma)

	return (normalized_X, mu, sigma)

def plotFit(min_x, max_x, mu, sigma, theta, p):
	x = np.arange(min_x - 15, max_x + 25, 0.05).reshape((-1,1))
	X_poly = polyFeatures(x,p)
	X_poly = np.divide(X_poly - mu, sigma)
	X_poly = np.hstack((np.ones(len(X_poly)).reshape((-1,1)),X_poly))
	plt.plot(x,np.dot(X_poly,theta),'b--',linewidth=2)
	return

def validationCurve(X, y, Xval, yval):
	lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape((-1,1))
	error_train = np.zeros(len(lambda_vec))
	error_val = np.zeros(len(lambda_vec))

	for i in range(len(lambda_vec)):
		m_train = len(X)
		m_val = len(Xval)
		reg_param = lambda_vec[i]
		est_theta = trainLinearReg(X,y,reg_param)
		error_train[i] = (np.sum((np.dot(X,est_theta)-y)**2))/(2.0*m_train)
		error_val[i] = (np.sum((np.dot(Xval,est_theta)-yval)**2))/(2.0*m_val)

	return (lambda_vec, error_train, error_val)




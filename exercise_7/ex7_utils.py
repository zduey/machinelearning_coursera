import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def findClosestCentroids(X, centroids):
	K = np.size(centroids, 1)
	idx = []

	for i in range(len(X)):
		norm = np.sum(((X[i] - centroids)**2), axis=1)
		idx.append(norm.argmin())
		
	return idx

def computeCentroids(X, idx, K):
	centroid = np.zeros((K,np.size(X,1)))
	aug_X = np.hstack((np.array(idx)[:,None],X))
	for i in range(K):
		centroid[i] = np.mean(X[aug_X[:,0] == i], axis=0)
	
	return centroid

def runKMeans(X, initial_centroids, max_iters, plot_progress=False):
	K = np.size(initial_centroids, 0)
	centroids = initial_centroids 
	previous_centroids = centroids

	for i in range(max_iters):
		# Centroid assignment
		idx = findClosestCentroids(X, centroids)

		if plot_progress:
			plt.plot(X[:,0],X[:,1], 'bo')
			plt.plot(centroids[:,0], centroids[:,1], 'rx')
			plt.plot(previous_centroids[:,0], previous_centroids[:,1], 'gx')
			plt.show()

			previous_centroids = centroids
			centroids = computeCentroids(X, idx, K)

	return (centroids, idx)

def kMeansInitCentroids(X, K):
	return X[np.random.choice(X.shape[0], K)]

def featureNormalize(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0)
	normalized_X = np.divide(X - mu,sigma)

	return (normalized_X, mu, sigma)

def pca(X):
	covar = np.dot(X.T,X) / len(X)
	U, S, V = np.linalg.svd(covar)
	return (U, S)

def projectData(X, U, K):
	U_reduce = U[:, 0:K]
	Z = np.zeros((len(X), K))
	for i in range(len(X)):
		x = X[i,:]
		projection_k = np.dot(x, U_reduce)
		Z[i] = projection_k
	return Z

def recoverData(Z, U, K):
	X_rec = np.zeros((len(Z), len(U)))
	for i in range(len(Z)):
		v = Z[i,:]
		for j in range(np.size(U,1)):
			recovered_j = np.dot(v.T,U[j,0:K])
			X_rec[i][j] = recovered_j
	return X_rec

def displayData(X):
    """
    Displays 2D data stored in design matrix in a nice grid.
    """
    num_images = len(X)
    rows = int(num_images**.5)
    cols = int(num_images**.5)
    fig, ax = plt.subplots(rows,cols,sharex=True,sharey=True)
    img_num = 0

    for i in range(rows):
        for j in range(cols):
            # Convert column vector into 32x232 pixel matrix
            # You have to transpose to have them display correctly
            img = X[img_num,:].reshape(32,32).T
            ax[i][j].imshow(img,cmap='gray')
            img_num += 1

    return (fig, ax)

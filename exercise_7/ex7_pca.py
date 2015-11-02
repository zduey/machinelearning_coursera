# Coursera Online Machine Learning Course
# Exercise 7 -- Principal Component Analysis and K-Means Clustering

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex7_utils import *
import scipy.io
import matplotlib.pyplot as plt

# Part 1 -- Load Example Dataset
raw_mat = scipy.io.loadmat("ex7data1.mat")
X = raw_mat.get("X")
plt.cla()
plt.plot(X[:,0], X[:,1], 'bo')
plt.show()

# Part 2 -- Principle Component Analysis
X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)

# Part 3 -- Dimension Reduction
plt.cla()
plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.show()

K = 1
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)

plt.cla()
plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.plot(X_rec[:,0], X_rec[:,1], 'rx')
plt.show()

# Part 4 -- Loading and Visualizing Face Data
raw_mat = scipy.io.loadmat("ex7faces.mat")
X = raw_mat.get("X")
face_grid, ax = displayData(X[:100, :])
face_grid.show()

X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)

face_grid, ax = displayData(U[:,:36].T)
face_grid.show()

# Part 6 -- Dimension Reduction on Faces
K = 100
Z = projectData(X_norm, U, K)

# Part 7 -- Visualization of Faces after PCA Dimension Reduction
K = 100
X_rec  = recoverData(Z, U, K)

plt.close()
plt.cla()
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
f, ax1 = displayData(X_norm[:100,:])
f, ax2 = displayData(X_rec[:100,:])
f.show()


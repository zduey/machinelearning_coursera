# Coursera Online Machine Learning Course
# Exercise 7 -- Principal Component Analysis and K-Means Clustering

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex7_utils import *
import scipy.io
import matplotlib.pyplot as plt

# Part 1 -- Find Closest Centroids
raw_mat = scipy.io.loadmat("ex7data2.mat")
X = raw_mat.get("X")

# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, initial_centroids)

# Part 2 -- Compute Means
centroids = computeCentroids(X, idx, K)

# Part 3 -- K-means Clustering
max_iters = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
centroids, idx = runKMeans(X, initial_centroids, max_iters, plot_progress=True)

# Part 4 -- K-means Clustering on Pixels
A = plt.imread("bird_small.png")
plt.imshow(A)
plt.show()

original_shape = np.shape(A)

# Reshape A to get R, G, B values for each pixel
X = A.reshape((np.size(A, 0)*np.size(A, 1), 3))
K = 16
max_iters = 10

# Initialize centroids
initial_centroids = kMeansInitCentroids(X, K)

# Run K-means
centroids, idx = runKMeans(X, initial_centroids, max_iters, plot_progress=False)

# Part 5 -- Image Compression
idx = findClosestCentroids(X, centroids)
X_recovered = centroids[idx,:]
X_recovered = X_recovered.reshape(original_shape)

# Display Images 
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.imshow(A)
ax2.imshow(X_recovered)

# Coursera Online Machine Learning Course
# Exercise 8 -- Anomaly Detection and Recommender Systems

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex8_utils import *
import scipy.io
import matplotlib.pyplot as plt

# Part 1 -- Load Example Data
raw_mat = scipy.io.loadmat("ex8data1.mat")
X = raw_mat.get("X")
Xval = raw_mat.get("Xval")
yval = raw_mat.get("yval")

plt.plot(X[:, 0], X[:, 1], 'bx')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)');
plt.show()

# Part 2 -- Estimate the dataset statistics
mu, sigma2 = estimateGaussian(X) # returns flattened arrays

# Density of data based on multivariate normal distribution
p = multivariateGaussian(X, mu, sigma2)

# Visualize the fit
fig, ax = visualizeFit(X,  mu, sigma2)
fig.show()

# Part 3 -- Find Outliers
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)

outliers = np.where(p < epsilon)
fig, ax = visualizeFit(X,  mu, sigma2)
ax.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
fig.show()

# Part 4 -- Multi-Dimensional Outliers
raw_mat2 = scipy.io.loadmat("ex8data2.mat")
X = raw_mat2.get("X")
Xval = raw_mat2.get("Xval")
yval = raw_mat2.get("yval")

mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)


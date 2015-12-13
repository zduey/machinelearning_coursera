# Coursera Online Machine Learning Course
# Exercise 8 -- Anomaly Detection and Recommender Systems

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex8_utils import *
import scipy.io
import matplotlib.pyplot as plt

# Part 1 -- Load Movie Ratings Dataset
raw_mat = scipy.io.loadmat("ex8_movies.mat")
R = raw_mat.get("R") # num movies x num users indicator matrix
Y = raw_mat.get("Y") # num movies x num users ratings matrix

# Visualize matrix
plt.matshow[.]
plt.xlabel("Users")
plt.ylabel("Movies")
plt.show()

# Part 2 -- Collaborative Filtering Cost Function
raw_mat2 = scipy.io.loadmat("ex8_movieParams.mat")
X = raw_mat2.get("X") # rows correspond to feature vector of the ith movie 
Theta = raw_mat2.get("Theta") # rows are the parameter vector for jth user

# Reduce data size to have it run faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# Evaluate Cost Function
params = np.append(X.flatten(), Theta.flatten())
J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
np.testing.assert_almost_equal(22.22, J,decimal=2, err_msg="Incorrect unregularized error")

# Part 3 -- Collaborative Filtering Gradient
checkCostFunction(0)

# Part 4 -- Collaborative Filtering Cost Regularization
J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
np.testing.assert_almost_equal(31.34, J,decimal=2, 
    err_msg="Incorrect regularized cost")

# Part 5 -- Collaborative Filtering Gradient Regularization
checkCostFunction(1.5)

# Part 6 -- Entering ratings for a new users
movieList = pd.read_table("movie_ids.txt",encoding='latin-1',names=["Movie"])
movies = movieList.Movie.tolist()
my_ratings = [0]*len(movies)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53]= 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68]= 5
my_ratings[182]= 4
my_ratings[225]= 5
my_ratings[354]= 5

for i in range(len(movies)):
    if my_ratings[i] > 0:
        print("User rated " + str(movies[i]) + ": " + str(my_ratings[i]))

# Part 8 -- Learning Movie Ratings
raw_mat = scipy.io.loadmat("ex8_movies.mat")
R = raw_mat.get("R") # num movies x num users indicator matrix
Y = raw_mat.get("Y") # num movies x num users ratings matrix

# Add own ratings to Y
ratings_col = np.array(my_ratings).reshape((-1,1))
Y = np.hstack((ratings_col, Y))

# Add indicators to R
R = np.hstack((ratings_col !=0, R))

# Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y,R)

# Useful values
num_users = np.size(Y,1)
num_movies = np.size(Y,0)
num_features = 10

# Set initial parameters
X = np.random.normal(size=(num_movies, num_features))
Theta = np.random.normal(size=(num_users, num_features))

initial_parameters = np.append(X.flatten(), Theta.flatten())
reg = 10

def reducedCofiCostFunc(p):
    """ Cheaply decorated cofiCostFunction """
    return cofiCostFunc(p,Y, R, num_users, num_movies, num_features,reg)

results = minimize(reducedCofiCostFunc,
                   initial_parameters,
		   method="CG",
                   jac=True,
                   options={'maxiter':100, "disp":True})

out_params = results.x

# Unfold the returned parameters back into X and Theta
X = np.reshape(out_params[:num_moves*num_features], (num_movies, num_features))
Theta = np.reshape(out_params[num_movies*num_features:],
    (num_users,num_features))

# Part 9 -- Recommendation for you
p = np.dot(X, Theta.T)
my_predictions = p[:,0] + Ymean.T.flatten()
sorted_predictions = np.sort(my_predictions)
sorted_ix = my_predictions.ravel().argsort()

print("\nTop recommendations for you:\n")
for i in range(10):
    j = sorted_ix[-i]
    print("Predicting rating " + str(my_predictions[j]) + 
	" for movie " + str(movies[j]))

print("\n Original ratings provided: \n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated " + str(my_ratings[i]) + " for " + str(movies[i]))

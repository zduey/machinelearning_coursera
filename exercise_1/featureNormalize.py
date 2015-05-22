#!/usr/bin/env python

import numpy as np

def featureNormalize(X):
    """
    Normalizes (mean=0, std=1) the features in design matrix X

    returns -- Normalized version of X where the mean of each
               value of each feature is 0 and the standard deviation
	       is 1. This will often help gradient descent learning
	       algorithms to converge more quickly.

    Instructions: First, for each feature dimension, compute the mean
                  of the feature and subtract it from the dataset,
		  storing the mean value in mu. Next, compute the 
		  standard deviation of each feature and divide
		  each feature by it's standard deviation, storing
		  the standard deviation in sigma. 
		  
		  Note that X is a matrix where each column is a 
		  feature and each row is an example. You need 
		  to perform the normalization separately for 
		  each feature. 
		  
		  Hint: You might find the 'mean' and 'std' functions useful.

    """
    
    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

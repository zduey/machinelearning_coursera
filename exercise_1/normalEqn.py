#!/usr/bin/env python

import numpy as np

def normalEqn(X,y):
    """
    Computes the closed form least squares solution using normal
    equations

    theta = (X^T*X)^{-1}X^T*y
    Returns: Array of least-squares parameters
    """
    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

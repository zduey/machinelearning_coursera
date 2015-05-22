#!/usr/bin/env python

import matplotlib.pyplot as plt

def plotData(x, y):
    """
    plotData -- Plots the data points x and y into a new figure and gives
                the figure axes labels of population and profit. It returns
		at matplotlib figure.

     Instructions: Plot the training data into a figure by manipulating the
		   axes object created for you below. Set the axes labels using
                   the "xlabel" and "ylabel" commands. Assume the 
                   population and revenue data have been passed in
                   as the x and y arguments of this function.
    
     Hint: You can use the 'rx' option with plot to have the markers
           appear as red crosses. Furthermore, you can make the
           markers larger by using plot(..., 'rx', markersize=10)
    """
    ########################### Your Code HERE ############################ 
    fig, ax = plt.subplots() # create empty figure and set of axes
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")

    return fig


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

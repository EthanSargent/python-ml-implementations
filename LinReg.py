# Author: Ethan Sargent
#
# The following is an implementation of multiple linear regression (for an
# arbitrary number of parameters) using gradient descent.
#
# In the example, we predict weight from blood pressure and age, and plot the
# decrease of the cost function over time to verify gradient descent is
# working.
#
# Retrieved dataset from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html

import csv
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, Y, iterations, alpha, l = 0):
    """ We use gradient descent the coefficients (betas) of a linear function
        of multiple variables. By default, l = 0; setting l > 0  will penalize
        large betas which corrects for overfitting, and this becomes
        regularized gradient descent.
    """
    
    # initialize B0, B1, ..., Bp
    betas = np.array([0.0]*(len(X[0])+1))
    
    # initialize list of cost vs iterations; should see a gradual descent
    costs = np.array([0.0]*iterations)
    
    # number of observations
    m = len(X)
    
    for i in range(iterations):
        sumterms = 1.0/m * ([estimation(xvec,betas) for xvec in X] - Y)
        errors = np.array([0.0]*len(betas))
        errors[0] = sum(sumterms) # error term for B0 has no multiplier
        for k in range(1,len(betas)):
            errors[k] = np.dot(sumterms, [row[k-1] for row in X])
            
        betas = betas - alpha * errors
        costs[i] = cost(X, Y, betas)
    
    return betas, costs

def estimation(xvec, betas):
    # B0 + B1*X1 + B2*X2 + ... + Bp * Xp
    return (betas[0] + np.dot(xvec, betas[1:]))
    
def cost(X, Y, betas):
    # the total cost for our data for some betas; higher cost indicates worse
    # performing betas and/or too large betas i.e. overfitting
    total_cost = 0
    
    for i in range(len(data)):
        total_cost += (estimation(X[i], betas) - Y[i])**2
        
    return total_cost/(2*len(X))
    
# column to predict
c = 2 # weight
    
# read data, take off row of labels (which evaluate to NaNs)
data = np.genfromtxt('BloodPressure.csv', delimiter = ',')[1:]

# delete the cth column from the data, which we will attempt to predict
X = np.delete(data, c, 1)
Y = np.array([row[c] for row in data])

# we use quite a low alpha; for alpha around .01-.001 errors diverge
# for this data set.
betas, costs = gradient_descent(X,Y, 1000, .00001)

# on this data set, cost descends incredibly quickly
plt.plot(range(len(costs)), costs)
plt.show()

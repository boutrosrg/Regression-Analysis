"""
This python script contains implementation of the multivariate linear regression algorithms described in the following post:
__article__ = Linear Regression "http://www.data-automaton.com/2017/12/25/linear-regression/"
__author__ = "Boutros El-Gamil"
__web__ = "www.data-automaton.com"
__github__ = "https://github.com/boutrosrg"
__email__ = "contact@data-automaton.com"
"""

import numpy as np
import matplotlib.pyplot as plt

def scale_linear_bycolumn(rawpoints, low=0.0, high=1.0):
    '''
    normalization function that sets each vector in range [0,1]
    
    INPUTS:
    :param rawpoints: vector of numerical values
    :param low: lower normalization value  
    :param high: higher normalization value
    
    OUTPUTS:
    :param rawpoints_normalized: vector of normalized numerical values  
    '''
    
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    
    rawpoints_normalized = high - (((high - low) * (maxs - rawpoints)) / rng)
    return rawpoints_normalized

def closed_form(x,y):
    ''' 
    implementation of closed form solution of linear regression using the equation W = X^-1 * Y
    INPUTS:
    :param x: array of independent variables (predictors)
    :param y: dependent variable    
    
    OUTPUTS:
    :param W: vector of coefficients of the linear function between x and x    
    '''       
    
    # number of observations    
    N = x.shape[0]
    # number of predictors
    M = x.shape[1]
    # initialize squared matrix of X of independent variables x    
    X = np.zeros((N,M+1))        
    
    # set first column equal to 1
    X[:,0] = 1
    
    # append x to X
    for c in range(1,X.shape[1]):
        X[:,c]= x[:,c-1]
        
    print X.shape
    print Y.shape
    
    # compute weight vector
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)               
    
    return W

def batch_gradient_descent(X, Y, W, eta, iterations):
    ''' 
    implementation of gradient descent algorithm to compute the coefficients of 
    the best linear function between 2 variables X and Y
    INPUTS:
    :param X: array of independent variables (predictors)
    :param Y: dependent variable
    :param W: vector of initial weights 
    :param eta: learning rate
    :param iterations: number of iterations applied on updates
    
    OUTPUTS:
    :param W: vector of coefficients of the linear function between X and Y
    :param error: (SSE) sum of differences between true and predicted values of dependent variable Y
    '''       
    
    # number of observations    
    N = X.shape[0]
    # number of predictors
    M = X.shape[1]    
    
    # update weights
    for k in range(0, iterations):
        # set gradient of weight vector to zeros
        W_gradient = np.zeros(len(W))                
                            
        # compute derivatives of weight vector        
        # loop over weight vector
        for m in range(0,len(W)):
                                              
            # loop over obsevations
            for i in range(0,N):
                                                
                # compute f(Xi,W)                
                # init f(Xi,W) to W_0
                f_xi_w = W[0]
                # add higher weights multiplied by corresponded predictors
                for j in range(0,M):                                                         
                    f_xi_w +=  W[j+1] * X[i,j]   
                    
                # get target observation # i                
                y = Y[i]                
                    
                if m == 0:
                    x_i_m = 1
                else:
                    x_i_m = X[i,m-1]
                    
                # compute derivative of coefficient m
                W_gradient[m] +=  x_i_m * (y - f_xi_w )
            
            # multiply derivative W_m by -2/N
            W_gradient[m] *= -2/N                    
                        
        # compute the updated weight vector    
        W_update = [W[0] - (eta * W_gradient[0]), W[1] - (eta * W_gradient[1]), W[2] - (eta * W_gradient[2])]
        
        # set weight vector to the updated one
        W = W_update
        
        print 'iter= ', k, ' W_update=' , W_update
        
    return W

if __name__ == '__main__':
    ## for this purpose, we use Housing data
    ## source (https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/Boston.html)
    # read data from csv file, ignoring col. names
    data = np.genfromtxt('boston.csv', delimiter=',')
    
    # convert data to float 
    data = np.array(data).astype(np.float) 
    
    # delete col. names (i.e. first row)
    data = np.delete(data, (0), axis=0)
                
    # Normalization
    data = scale_linear_bycolumn(data)    
    
    # define independent variables X1, X2, dependent variable Y 
    # X1: lower status of the population (percent).
    # X2: average number of rooms per dwelling.
    # Y: median value of owner-occupied homes in \$1000s.
    X1, X2, Y = data[:,12], data[:,5], data[:,13]  
    
    # append predictor variables to X    
    X = np.append([X1], [X2], axis=0)   
    
    # transpose array X such that dimensions become (observation x feature)
    X = X.T    
    
    ## 1- closed-form method
    W_cf = closed_form(X,Y)
    print "Closed Form W: ", W_cf
    
    ## 2- gradient descent method
    W_gd = batch_gradient_descent(X, Y, [0,0,0], .00001, 100)   
    print "Gradient Descent W: ", W_gd
    
    
    ## 3. plotting
    # set m to either 0 or 1 to plot Y against either first or second predictors
    m= 1
    plt.scatter(X[:,m], Y)
    plt.xlabel('x', fontweight='bold')    
    plt.ylabel('y', fontweight='bold')         
    
    ## fit your data
    yfit_cf = np.zeros(X.shape[0])
    yfit_gd = np.zeros(X.shape[0])
    
    for i in range(0, X.shape[0]):        
        yfit_cf[i] = W_cf[0] +  (W_cf[1] * X[i,0]) + (W_cf[2] * X[i,1])
        yfit_gd[i] = W_gd[0] +  (W_gd[1] * X[i,0]) + (W_gd[2] * X[i,1])
               
    print "Closed Form SSE= ", np.sum(np.power([yfit_cf - Y],2))
    print "Gradient Descent SSE= ", np.sum(np.power([yfit_gd - Y],2))
    
    plt.plot(X[:,m], yfit_gd, color= 'green', label='Grad. Descent')
    plt.plot(X[:,m], yfit_cf, color= 'red', label='Closed Form')  
    
    leg= plt.legend(loc="upper left")
    
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
        
    plt.title("MV Linear Regression", fontweight='bold')
    plt.show()
    
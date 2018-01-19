"""
This python script contains implementation of the linear regression algorithms described in the following post:
__article__ = Linear Regression "http://www.data-automaton.com/2017/12/25/linear-regression/"
__author__ = "Boutros El-Gamil"
__web__ = "www.data-automaton.com"
__github__ = "https://github.com/boutrosrg"
__email__ = "contact@data-automaton.com"
"""

import numpy as np
import matplotlib.pyplot as plt

def batch_gradient_descent(X, Y, W, eta, iterations):
    ''' 
    implementation of gradient descent algorithm to compute the coefficients of 
    the best linear function between 2 variables X and Y
    INPUTS:
    :param X: independent variable
    :param Y: dependent variable
    :param W: vector of initial weights 
    :param eta: learning rate
    :param iterations: number of iterations applied on updates
    
    OUTPUTS:
    :param W: vector of coefficients of the linear function between X and Y
    :param error: (SSE) sum of differences between true and predicted values of dependent variable Y
    '''       
    
    # get number of obserations
    N = float(len(X))    
    
    # update weights
    for j in range(0, iterations):
        # set gradient of weight vector to zeros
        W_gradient = np.zeros(len(W))                
        
        # loop over obsevations
        for i in range(0, len(X)):
            # get coordinates of observation # i
            x = X[i]
            y = Y[i]
            
            # compute derivatives of weight vector
            W_gradient[0] += (2/N) * ((W[1] * x) + W[0] - y)
            W_gradient[1] += (2/N) * x * ((W[1] * x) + W[0] - y)
                        
        # compute new weight vector    
        W_new = [W[0] - (eta * W_gradient[0]), W[1] - (eta * W_gradient[1])]
        
        # set weight vector to the updated one
        W = W_new
        
        ## compute total error        
        # initialize total error
        totalError = 0
        
        # loop over obsevations
        for i in range(0, len(X)):
            # get coordinates of observation # i
            x = X[i]
            y = Y[i]        
            totalError += (y - (W[1] * x + W[0])) ** 2
    
        print 'j= ', j, ' W_gradient=' , W_gradient , ' ERROR=' , totalError
    return W, totalError

def closed_form(x,y,p):
    ''' 
    implementation of closed form solution of linear regression using the equation W = X^-1 * Y
    INPUTS:
    :param x: independent variable
    :param y: dependent variable
    :param p: polinomial degree     
    
    OUTPUTS:
    :param W: vector of coefficients of linear regression between x and y    
    '''                 
    
    # initialize squared matrix of X of independent variable x    
    X = np.zeros((p+1,p+1))
    
    # initialize vector y of independent variable x multiplied by dependent variable y 
    Y = np.zeros((p+1))
    
    # build X, Y matrices
    for k in range (0,p+1):
        Y[[k]] = np.sum(np.power(x,k).dot(y))
        row = k
        col = 0
        for j in range (k,p+k+1):
            X[row,col] = j
            X[row,col] = np.sum(np.power(x,j))
            col+= 1
    
    # insert number of observation at position (1,1) of matrix X        
    X[0,0] = len(x)
           
    # solve equation W = X^-1 * Y to obtain weights vector W
    W = np.linalg.inv(X).dot(Y)    
    
    ## compute total error    
    # initialize total error
    totalError = 0
    
    # loop over obsevations
    for i in range(0, len(x)):
        # get coordinates of observation # i
        x_curr = x[i]
        y_curr = y[i]        
        totalError += (y_curr - (W[1] * x_curr + W[0])) ** 2
        
    return W, totalError


if __name__ == '__main__':
    
    # define data array
    data = np.array([[65.78, 71.52, 69.40, 68.22, 67.79, 68.70, 69.80, 70.01, 67.90, 66.78, 66.49, 67.62, 68.30, 67.12, 68.28, 71.09, 66.46, 68.65, 71.23, 67.13],
                    [112.99, 136.49, 153.03, 142.34, 144.30, 123.30, 141.49, 136.46, 112.37, 120.67, 127.45, 114.14, 125.61, 122.46, 116.09, 140.00, 129.50, 142.97, 137.90, 124.04]])
    
    # define independent variable X, dependent variable Y 
    X, Y = data[0,:], data[1,:]    
    
    ## 1- gradient descent method
    W_gd, error_gd = batch_gradient_descent(X, Y, [1,1], .00001, 2000)        
    
    print 'Gradient Descent W: ', W_gd    
    print 'Gradient Descent ERROR: ', error_gd
    
    ## 2- closed-form method    
    W_cf, error_cf = closed_form(X, Y, 1)
    
    print 'Closed Form W: ', W_cf    
    print 'Closed Form ERROR: ', error_cf
    
    ## 3. plotting
    plt.scatter(X, Y)
    plt.xlabel('x', fontweight='bold')    
    plt.ylabel('y', fontweight='bold')     
    
    axes = plt.gca()
    axes.set_xlim([65,72]) 
    axes.set_ylim([110,160])    
    
    ## fit your data
    yfit_gd = [W_gd[0] + W_gd[1] * xi for xi in X]
    yfit_cf = [W_cf[0] + W_cf[1] * xi for xi in X]
    
    plt.plot(X, yfit_gd, color= 'green', label='Grad. Descent')
    plt.plot(X, yfit_cf, color= 'red', label='Closed Form')
    leg= plt.legend(loc="upper left")
    
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
        
    plt.title("Linear Regression", fontweight='bold')
    plt.show()
               
"""
This python script contains implementation of the multivariate logistic regression methods described in the following post:
__article__ = Logistic Regression ""
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
    logistic regression function between 2 variables X and Y
    INPUTS:
    :param X: independent variable
    :param Y: dependent variable
    :param W: vector of initial weights 
    :param eta: learning rate
    :param iterations: number of iterations applied on updates
    
    OUTPUTS:
    :param W: vector of coefficients of the logistic function between X and Y    
    '''       
    
    # get number of obserations
    N = float(len(X))    
    
    # update weights
    for k in range(0, iterations):
        
        # set gradient of weight vector to zeros
        W_gradient = np.zeros(len(W))   
        
        # set new weight vector
        W_new = np.zeros(len(W))  
        
        # loop over obsevations
        for i in range(0, len(X)):
            
            # get coordinates of observation # i
            x = X[i,:]
            y = Y[i]
            
            # compute logistic function g(x,w)            
            g_w_x = 1/(1 + np.exp(-((W[1:].dot(x)) + W[0])))
            
            # compute logistic hypothesis h(w)
            h_w = (y * g_w_x) + ((1-y) * g_w_x)
            
            # compute derivatives of weight vector            
            W_gradient[0] +=  y - h_w 
            
            for j in range (0,len(x)):
                W_gradient[j+1] +=  x[j] * (y - h_w)
        
        
        # multiply derivative W_m by -1/N
        W_gradient *= -1/(N)    
        ## or reverse for gradient ascent
        #W_gradient *= 1/N
        
        # compute new weight vector  
        for j in range (0,len(W)):
            W_new[j] = W[j] - (eta * W_gradient[j])
        
        ## or reverse for gradient ascent
        #for j in range (0,len(W)):
        #    W_new[j] = W[j] + (eta * W_gradient[j])
        
        # set weight vector to the updated one
        W = W_new        
        
        print "Gradient Descent: Iter # ", k,  " W= ", W
        
    return W

def newton_method(X, Y, W, iterations):
    ''' 
    implementation of Newton's method to compute the coefficients of 
    logistic regression function between 2 variables X and Y
    INPUTS:
    :param X: independent variable
    :param Y: dependent variable
    :param W: vector of initial weights 
    :param iterations: number of iterations applied on updates
    
    OUTPUTS:
    :param W: vector of coefficients of the logistic function between X and Y    
    '''       
    
    # get number of obserations
    N = float(len(X))    
    
    # update weights
    for k in range(0, iterations):
        
        # set gradient of weight vector to zeros
        W_gradient = np.zeros(len(W))                
        
        ## add vector of logistic function g(x,w) values
        G_w_x = np.zeros(len(X)) 
        
        # loop over obsevations
        for i in range(0, len(X)):
            
            # get coordinates of observation # i
            x = X[i,:]
            y = Y[i]
            
            # compute logistic function g(x,w)
            g_w_x = 1/(1 + np.exp(-((W[1:].dot(x)) + W[0])))
            
            # compute logistic hypothesis h(w)
            h_w = (y * g_w_x) + ((1-y) * g_w_x)                        
            
            # compute derivatives of weight vector            
            W_gradient[0] +=  y - h_w 
            
            for j in range (0,len(x)):
                W_gradient[j+1] +=  x[j] * (y - h_w)
        
            ## add g_w_x values to G_w_x
            G_w_x[i] = g_w_x                    
        
        # multiply derivative W_m by -2/N
        W_gradient *= -1/(N)            
        
        ## append ones column to independent vector(s) X
        ones = np.ones(len(X))        
        X2 = np.column_stack((ones,X)) 
        
        ## compute Hessian Matrix
        H = (-1/N) * X2.T.dot(np.diag(G_w_x)).dot(1-np.diag(G_w_x)).dot(X2)    
        
        # compute new weight vector 
        W_new = (np.linalg.inv(H).dot(W_gradient))        

        # set weight vector to the updated one
        W = W_new         
        
        print "Newton's Method: Iter # ", k,  " W= ", W
        
    return W

def compute_predictions(X, W):
    '''
    make predictions of binary logistic regression by generatic the probability that each observation is of class 1.
    If the probability is >= 0.5, then observation is classified as '1', otherwise as '0' 
    
    INPUTS:
    :param X: independent variable (predictor)
    :param W: vector of estimated weights (coefficients)
    
    OUTPUTS:
    :param y_fit: vector of predictions to target variable
    :param rmse: Root Mean Square Error of the predictions
    '''
    
    # set exponential component of logit 
    exp_component = np.zeros(len(X)) 
    
    # compute exponential component of logit 
    for i in range(0,len(X)):
        exp_component[i] = np.exp(-((W[1:].dot(X[i,:])) + W[0]))         
    
    # compute the estimated probabilities    
    y_fit =  [e/(1 + e) for e in exp_component] 
    
    return y_fit

def compute_RMSE(Y, y_fit):
    '''
    Compute the root mean square error of logistic regression between vector of true classes (Y) and vector of estimated classes (y_fit) 
    
    INPUTS:
    :param Y: dependent (target) variable
    :param y_fit: vector of predictions to target variable
    
    OUTPUTS:
    :param rmse: Root Mean Square Error of the predictions
    '''
    
    # initialize total error
    totalError = 0                

    # loop over obsevations
    
    for i in range(0, len(Y)):          
        # compute error rate
        totalError += (Y[i] - round(y_fit[i])) ** 2    
        
    return np.sqrt(totalError / len(Y))

def plot_regression(x, y, y_fit, title):
    '''
    plot fitted curves of logistic regression algorithms
    INPUTS:
    :param x: independent variable
    :param y: dependent variable
    :param y_fit: fitted curve of dependent variable    
    :param title: figure title        
    '''    
    
    plt.scatter(x,y)
    plt.xlabel('x', fontweight='bold')    
    plt.ylabel('y', fontweight='bold')             
        
    plt.scatter(x, y_fit, linewidth=2, color= 'red')            
        
    plt.title(title, fontweight='bold')    

    plt.show()  

if __name__ == '__main__':
    
    ## read data from csv file, ignoring col. names       
    data = np.genfromtxt('data.csv', delimiter=',')
    
    # predictor variabless X (2 predictors)
    X = data[:,1:3]
    
    # target variable y 
    Y = data[:,0]  
    
    ## estimate weights (gradient descent)
    W_init = np.array([10, 1, 1])
    eta = 0.0005
    iterations = 7000    
    
    W_gd = batch_gradient_descent(X, Y, W_init, eta, iterations)
    
    ## compute predictions
    y_gd = compute_predictions(X, W_gd)  
    
    ## compute errors
    error = compute_RMSE(Y, y_gd)
    print "Gradient Descent ERROR= ", error 
    
    ## plotting    
    plot_regression(X[:,1], Y, y_gd, "Logistic Regression (Gradient Descent)")    
    
    
    ## estimate weights (Newton's method)
    W_init = np.array([10, 1, 1])
    iterations= 500
    W_nm = newton_method(X, Y, W_init, iterations)
    print "Newton's method W= ", W_nm  
         
    ## compute predictions
    y_nm = compute_predictions(X, W_nm)
    
    ## compute errors
    error = compute_RMSE(Y, y_nm)
    print "Newton's method ERROR= ", error
    
    ## plotting    
    plot_regression(X[:,1], Y, y_nm, "Logistic Regression (Newton's Method)")
    
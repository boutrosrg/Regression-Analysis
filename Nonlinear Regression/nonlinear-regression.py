"""
This python script contains implementation of the nonlinear regression algorithms described in the following post:
__article__ = Nonlinear Regression "http://www.data-automaton.com/2018/01/09/nonlinear-regression/"
__author__ = "Boutros El-Gamil"
__web__ = "www.data-automaton.com"
__github__ = "https://github.com/boutrosrg"
__email__ = "contact@data-automaton.com"
"""

import numpy as np
import matplotlib.pyplot as plt

def closed_form(x,y,p):
    ''' 
    implementation of closed form solution of linear regression using the equation W = X^-1 * Y
    INPUTS:
    :param x: independent variable
    :param y: dependent variable
    :param p: polinomial degree     
    
    OUTPUTS:
    :param W: vector of coefficients of nonlinear regression between x and y    
    ''' 
    # get number of obserations
    N = len(y)
    
    # init A and B matrices    
    A = np.zeros((p+1,p+1))
    B = np.zeros((p+1))    

    # build A and B matrices
    for k in range (0,p+1):
        # compute B matrix
        B[[k]] = np.sum(np.power(x,k).dot(y))
        row = k
        col = 0
        for j in range (k,p+k+1):            
            A[row,col] = np.sum(np.power(x,j))
            col+= 1
    
        A[0,0] = N    
    
    # compute vector of weights    
    W= np.linalg.inv(A).dot(B)
    
    return W

def closed_form_regularized(x,y,p,lamda):
    ''' 
    implementation of closed form solution of linear regression using the equation W = [X^(T)*X − λ*I]^(−1) X^(T)*Y
    INPUTS:
    :param x: independent variable
    :param y: dependent variable
    :param p: polinomial degree 
    :param lamda: regularization coefficient
    
    OUTPUTS:
    :param W: vector of coefficients of nonlinear regression between x and y    
    ''' 
    
    # get number of obserations
    N = len(y)
    
    # init X matrix    
    X = np.zeros((N,p+1))      
    
   # build design matrix
    for k in range(0,p+1):
        X[:,k]= np.power(x,k)  
    
    # compute vector of weights
    I = np.identity(p+1)
    W = np.linalg.inv(X.T.dot(X) - (I * lamda)).dot(np.transpose(X)).dot(y)

    return W



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
    N = len(X)  

    # update weights
    for j in range(0, iterations):
        # set gradient of weight vector to zeros
        W_gradient = np.zeros(len(W))                
        
        # compute derivatives of weight vector        
        # loop over weight vector
        for m in range(0,len(W)):                        
            
            # loop over obsevations
            for i in range(0,N):
                # get coordinates of observation # i
                x = X[i]
                y = Y[i]
                
                # compute f(Xi,W)      
                f_xi_w = 0
                # add higher weights multiplied by corresponded predictors
                for p in range (0, len(W)):                                                         
                    f_xi_w +=  W[p] * np.power(x,p) 
                
                # compute the free term                 
                x_i_m = np.power(x,m)                    
                
                # compute derivative of coefficient W_m
                W_gradient[m] += (y - f_xi_w ) * x_i_m
                
            # multiply derivative W_m by -2/N
            W_gradient[m] *= -2/N                              
                        
        # compute new weight vector  
        
        # set gradient of weight vector to zeros
        W_new = np.zeros(len(W)) 
        
        for p in range (0, len(W)): 
            W_new[p] = W[p] - (eta * W_gradient[p])                        
                
        # set weight vector to the updated one
        W = W_new
                
    return W

def batch_gradient_descent_regularized(X, Y, W, eta, lamda, iterations):
    ''' 
    implementation of gradient descent algorithm to compute the coefficients of 
    the best linear function between 2 variables X and Y
    INPUTS:
    :param X: independent variable
    :param Y: dependent variable
    :param W: vector of initial weights 
    :param eta: learning rate
    :param lamda: regularization coefficient
    :param iterations: number of iterations applied on updates
    
    OUTPUTS:
    :param W: vector of coefficients of the linear function between X and Y
    :param error: (SSE) sum of differences between true and predicted values of dependent variable Y
    '''       
    
    # get number of obserations
    N = len(X)  

    # update weights
    for j in range(0, iterations):
        # set gradient of weight vector to zeros
        W_gradient = np.zeros(len(W))                
        
        # compute derivatives of weight vector        
        # loop over weight vector
        for m in range(0,len(W)):                        
            
            # loop over obsevations
            for i in range(0,N):
                # get coordinates of observation # i
                x = X[i]
                y = Y[i]
                
                # compute f(Xi,W)      
                f_xi_w = 0
                # add higher weights multiplied by corresponded predictors
                for p in range (0, len(W)):                                                         
                    f_xi_w +=  W[p] * np.power(x,p) 
                
                # compute the free term                 
                x_i_m = np.power(x,m)                    
                
                # compute derivative of coefficient W_m
                W_gradient[m] += (y - f_xi_w ) * x_i_m
                
            # multiply derivative W_m by -2/N
            W_gradient[m] *= -2/N  
            
            # add penalty term to the gradient, excluding intercept coefficient W[0]
            if(m > 0):
                W_gradient[m] += lamda * W[m]
                        
        ## compute new weight vector          
        # set gradient of weight vector to zeros
        W_new = np.zeros(len(W)) 
        
        for p in range (0, len(W)): 
            W_new[p] = W[p] - (eta * W_gradient[p])                        
                
        # set weight vector to the updated one
        W = W_new
                
    return W

def get_fitted_curve(x, p, W):
    '''
    compute the fitted curve by multiplying coefficients vector W by design matrix X
    INPUTS:
    :param x: independent variable
    :param p: polinomial degree
    :param W: coefficients vector
    
    OUTPUTS:
    :param fitted_curve: values of nonlinear regression function f(W,x)   
    '''
    
    # init design matrix
    X= np.zeros((len(x),p+1))
        
    # build design matrix
    for k in range(0,p+1):
        X[:,k]= np.power(x,k)
    
    # compute the fitted curve        
    fitted_curve = X.dot(W)
    
    return fitted_curve

def plot_regression(x, y, y_fit, y_fit_reg, title, y_fit_label, y_fit_reg_label):
    '''
    plot fitted curves of nonlinear regression algorithms
    INPUTS:
    :param x: independent variable
    :param y: dependent variable
    :param y_fit: fitted curve of dependent variable
    :param y_fit_reg: regularized fitted curve of dependent variable
    :param title: figure title
    :param y_fit_label: fitted curve legend title
    :param y_fit_reg_label: regularized fitted curve legend title
    '''    
    
    plt.scatter(x,y)
    plt.xlabel('x', fontweight='bold')    
    plt.ylabel('y', fontweight='bold')     
    
    axes = plt.gca()
    axes.set_ylim([-1.5,1.5])       
        
    plt.plot(x, y_fit, linewidth=2, color= 'green',  label= y_fit_label)
    plt.plot(x, y_fit_reg, linewidth=2, color= 'red', label= y_fit_reg_label)
    leg= plt.legend(loc="upper left")
    
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
        
    plt.title(title, fontweight='bold')    
   
    plt.show()    

if __name__ == '__main__':
    
    # number of data observations
    n_samples = 10
    
    # predictor variable x
    x= np.linspace(-1,1,n_samples)
    
    # target variable y (using sine function)
    y= np.zeros(len(x))
    for i in range (0,len(x)):
        y[i]= np.sin(3*x[i])  
    
    # add noise to target variable
    y+= np.random.randn(n_samples) * 0.1
    
    # set polynomial degree
    p= 3
    
    ## closed form
    W_cf = closed_form(x,y,p)    
    print 'Closed Form W= ', W_cf    
        
    ## closed form regularized
    lamda= 0.05

    W_cf_reg = closed_form_regularized(x,y,p,lamda)      
    print 'Closed Form (Regularized) W= ', W_cf_reg
                
    ## fit your data
    y_cf= get_fitted_curve(x, p, W_cf)  
    y_cf_reg= get_fitted_curve(x, p, W_cf_reg)         
    
    ## 3. plotting
    plot_regression(x, y, y_cf, y_cf_reg, "Nonlinear Regression (Closed Form)", 'Closed Form', 'Closed Form (Reg.)')
    
    ## gradient descent       
    W_init = np.zeros(p+1)
    eta = 0.0005
    iterations = 10000
    W_gd =  batch_gradient_descent(x, y, W_init, eta, iterations)
    print 'Gradient Descent W= ', W_gd
    
    ## gradient descent regularized
    lamda = 0.05
    W_gd_reg =  batch_gradient_descent_regularized(x, y, W_init, eta, lamda, iterations)
    print 'Gradient Descent (Regularized) W= ', W_gd_reg  
    
    ## fit your data
    y_gd= get_fitted_curve(x, p, W_gd)  
    y_gd_reg= get_fitted_curve(x, p, W_gd_reg)         
    
    ## 3. plotting
    plot_regression(x, y, y_gd, y_gd_reg, "Nonlinear Regression (Gradient Descent)", 'Gradient Descent', 'Gradient Descent (Reg.)')    
    
       
    

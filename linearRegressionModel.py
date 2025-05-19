import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import copy
import math

def compute_cost(x, y, w, b):
    """    
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """

    #number of Training Examples
    m = x.shape[0]

    #Cost to return
    total_cost = 0
    cost_sum = 0

    for i in range(m):
        xi = x[i][0]  # Extract scalar
        f_wb = w * xi + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    total_cost = cost_sum / (2*m)
    return total_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    
    """
    m = x.shape[0]

    dj_dw = 0.0
    dj_db = 0.0

    for i in range(m):
        xi = x[i][0]  # Extract scalar
        f_wb = w * xi + b
        error = f_wb - y[i]
        dj_dw += error * xi
        dj_db += error
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing




data = pd.read_csv("Delhi.csv")
x_train = data[["Area"]].values
x_mean = np.mean(x_train)
x_std = np.std(x_train)
x_train = (x_train - x_mean) / x_std

y_train = data["Price"]

w,b,_,_ = gradient_descent(x_train ,y_train, 0.0, 0.0, 
                     compute_cost, compute_gradient, 0.01, 1500)
print("w,b found by gradient descent:", w, b)

a,b = compute_gradient(x_train, y_train, 0.2 , 0.2)

cost = compute_cost(x_train, y_train, 2, 1)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

plt.plot(x_train, predicted, c = 'b')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Price in Delhi")
plt.xlabel("Area in sq. ft.")
plt.ylabel("Price")
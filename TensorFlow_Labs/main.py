import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('E:/development/deepLearning/diabetes.csv')
X = data[['Glucose','Insulin']].values
Y = data['Outcome'].values

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out

def sequential(x, W1, W2, B1, B2):
    a1 = my_dense(x, W1, B1)
    a2 = my_dense(a1, W2, B2)
    return a2
"""
Trained Weights from already performed algorithms

"""
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def my_predict(X, W1, W2, B1, B2):
    m = X.shape[0]
    p = np.zeros(m,1)
    for i in range(m):
        p[i,0] = sequential(X[i], W1, W2, B1, B2)
    return (p)

X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
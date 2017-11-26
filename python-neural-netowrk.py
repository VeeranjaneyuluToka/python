# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:20:21 2017

@author: Veeranjaneyulu Toka
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)
X, y = datasets.make_moons(200, noise = 0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

epislon = 0.01
reg_lambda = 0.01

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #Forward propogation to cal our predictions
    Z1 = X.dot(W1)+b1
    a1 = np.tanh(Z1)
    Z2 = a1.dot(W2)+b2
    exp_scores = np.exp(Z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims = True)
    #cal the loss
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    #add regulization term to loss
    data_loss += reg_lambda/2*(np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #forward propogation
    Z1 = x.dot(W1)+b1
    a1 = np.tanh(Z1)
    Z2 = a1.dot(W2)+b2
    exp_scores = np.exp(Z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim)/np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim)/np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    model={}
    
    for i in range(0, num_passes):        
        #Forward propogation
        Z1 = X.dot(W1)+b1
        a1 = np.tanh(Z1)
        Z2 = a1.dot(W2)+b2
        exp_scores = np.exp(Z2)
        probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        
        #Backword propogation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        dW2 += reg_lambda *W2
        dW1 += reg_lambda *W1
        
        W1 += -epislon*dW1
        b1 += -epislon*db1
        W2 += -epislon*dW2
        b2 += -epislon*db2
        
        model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
        
        if print_loss and i%1000 == 0:
            print("loss after iteration %i: %f"%(i, calculate_loss(model)))
    return model
    
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def hidden_layer_size_3():
    model = build_model(3, print_loss=True)
    plot_decision_boundary(lambda x: predict(model, x))
    plt.title("Decision boundary for hidden layer size 3")
    
def varying_hidden_layers():
    plt.figure(figsize=(16, 32))
    hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden layer size %d' % nn_hdim)
        model = build_model(nn_hdim)
        plot_decision_boundary(lambda x: predict(model, x))        
    plt.show()
    
def main():
    #hidden_layer_size_3()
    varying_hidden_layers()

if __name__ == "__main__":
    main()



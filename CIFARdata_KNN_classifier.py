# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:00:20 2018

@author: Veeranjaneyulu Toka
"""

import os
import pickle
import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.Xtr = X
        self.Ytr = y
        
    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)
        
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1) #L1 norm
#            distances = np.sqrt(np.sum(np.square(self.Xtr-X[i,:]), axis=1)) #L2 norm
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]
        return Ypred
    
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' %(b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def main():
    X_train, Y_train, X_test, Y_test = load_CIFAR10('data/cifar-10-batches-py/')
    
    X_train_rows = X_train.reshape(X_train.shape[0], 32*32*3)
    X_test_rows = X_test.reshape(X_test.shape[0], 32*32*3)
    
    #validation variables
    X_val_rows = X_train_rows[:1000, :]
    Yval = Y_train[:1000]
    Xtr_rows = X_train_rows[1000:, :]
    Ytr = Y_train[1000:]
    
    #find hyperparameters that work best
    validation_accuracies = []
    for k in [1, 3, 5, 10, 20, 50, 100]:
        nn = nearestNeighbor()
        nn.train(Xtr_rows, Ytr)
        
        Yval_predict = nn.predict(X_val_rows, k= k)
        acc = np.mean(Yval_predict==Yval)
        print('Accuracy : %f", acc)
        
        validation_accuracies.append((k, acc))
    
    nn = NearestNeighbor()
    nn.train(X_train_rows, Y_train)
    Y_test_predict = nn.predict(X_test_rows)
    
    print("accuracy:%f" %(np.mean(Y_test_predict == Y_test)))
    
    
if __name__ == "__main__":
    main()

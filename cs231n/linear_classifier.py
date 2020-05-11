
import numpy as np

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 #delta = 1
            if margin > 0:
                loss += margin

                dw = dW[:, y[i]].squeeze()
                dw = dw - X[i]
                dW[:, j] = dW[:, j] + X[i]

    loss /= num_train
    dW = dW / num_train

    loss += reg*np.sum(W*W)
    dW = dW + reg*2*W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dw = np.zeros(W.shape) #initialize the gradient as zero

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)

    correct_class_scores = scores[np.arange(num_train), y.squeeze()].reshape(num_train, 1)

    margin = np.maximum(0, scores - correct_class_scores + 1)

    margin[np.arange(num_train), y] = 0

    loss = margin.sum()/num_train

    #add regularization to the loss
    loss += reg*np.sum(W*W)


    #compute gradients
    margin[margin>0] = 1
    valid_margin_count = margin.sum(axis=1)

    #subtract incorrect class
    margin[np.arange(num_train), y.squeeze()] -= valid_margin_count
    dw = dw + reg*2*W

    return loss, dw


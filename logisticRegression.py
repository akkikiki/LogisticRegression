# coding: utf-8
import numpy as np
from math import log

def sigmoid(x, theta):
    return 1.0 / (1.0 + np.exp(-np.dot(theta.transpose(), x)))

def cost(x, y):
    return -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))

class LogisticRegression():
    """ Training is done using gradient descent
        See http://cs229.stanford.edu/notes/cs229-notes1.pdf for how the gradient is computed.
    """
    
    def __init__(self, X, Y):
        self.d = len(X[1,:])
        self.N = len(X[:,1])
        self.X_train = X
        self.Y_train = Y

    def update_matrix(self, thetas, alpha=0.1):
        new_thetas = np.zeros((self.d))
        for i in range(self.N):
            new_thetas = thetas + alpha * (self.Y_train[i] - sigmoid(self.X_train[i,:], thetas)) * self.X_train[i,:]

        return new_thetas


    def update(self, thetas, alpha=0.1):
        new_thetas = np.zeros((self.d))
        for i in range(self.N):
            for j in range(thetas.size):
                new_thetas[j] = thetas[j] - alpha * (sigmoid(self.X_train[i, :], thetas) - self.Y_train[i]) * self.X_train[i, j]

        return new_thetas

    def train(self, alpha=0.1):
        thetas = np.zeros(self.d)
        new_thetas = thetas + 1
        while thetas.all() != new_thetas.all():
            #tmp_new_thetas = self.update(old_thetas)
            tmp_new_thetas = self.update_matrix(thetas)
            thetas = new_thetas
            new_thetas = tmp_new_thetas
        return new_thetas


if __name__ == '__main__':
    X_train = np.array([[1, 2, 3], 
                        [2, 4, 8], 
                        [2, 4, 8], 
                        [2, 2, 2], 
                        [2, 5, 9]])
    Y_train = np.array([[1],
                        [1], 
                        [1], 
                        [0], 
                        [1]])

    LR = LogisticRegression(X_train, Y_train)
    #d = 3 dimension of each data
    #N = 5 num. of training data
    print LR.train()

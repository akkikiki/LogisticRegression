# coding: utf-8
import numpy as np
from math import log

def sigmoid(x, theta):
    #print x 
    #print theta
    return 1.0 / (1.0 + np.exp(-np.dot(theta.transpose(), x)))

def cost(x, y):
    return -y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))

class Data():
    def __init__(self):
        """ N number of training data with dimension d
        """

        self.X_train = np.zeros((N, d))
        self.Y_train = np.zeros(N) # N training examples

        self.X_test = np.zeros((N, d))
        self.Y_test = np.zeros(N)

class LogisticRegression():
    """ Training is done using gradient descent
        See http://cs229.stanford.edu/notes/cs229-notes1.pdf for how the gradient is computed.
    """

    def update_matrix(self, thetas, x, y, d, N, alpha=0.1):
        new_thetas = np.zeros((d))
        for i in range(N):
            new_thetas = thetas + alpha * (y[i] - sigmoid(x[i,:], thetas)) * x[i,:]

        return new_thetas


    def update(self, thetas, x, y, d, N, alpha=0.1):
        new_thetas = np.zeros((d))
        for i in range(N):
            for j in range(thetas.size):
                new_thetas[j] = thetas[j] - alpha * (sigmoid(x[i, :], thetas) - y[i]) * x[i, j]

        return new_thetas

    def train(self, thetas, x, y, d, N, alpha=0.1):
        old_thetas = thetas
        new_thetas = old_thetas + 1
        while old_thetas.all() != new_thetas.all():
            tmp_new_thetas = self.update(old_thetas, x, y, d, N)
            #tmp_new_thetas = self.update_matrix(old_thetas, x, y, d, N)
            old_thetas = new_thetas
            new_thetas = tmp_new_thetas
            #print new_thetas
        return new_thetas


if __name__ == '__main__':
    LR = LogisticRegression()
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
    d = 3
    N = 5
    thetas = np.zeros(d)
    
    print LR.train(thetas, X_train, Y_train, d, N)
    

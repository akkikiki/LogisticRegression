# coding: utf-8
import numpy as np
from math import log

def sigmoid(x, theta):
    return 1.0 / (1.0 + np.exp(-np.dot(theta.transpose(), x)))

class LogisticRegression():
    """ Training is done using gradient descent
        See http://cs229.stanford.edu/notes/cs229-notes1.pdf for how the gradient is computed.
    """
    
    def __init__(self, X, Y):
        """ N: num. of training data
            d: dimention of features
        """
        self.d = len(X[1,:])
        self.N = len(X[:,1])
        self.X_train = X
        self.Y_train = Y

    def update_matrix(self, thetas, alpha=0.1, C=1):
        """ C: regularization parameter
        """
        new_thetas = np.zeros((self.d))
        for i in range(self.N):
            new_thetas = thetas + alpha * (self.Y_train[i] - sigmoid(self.X_train[i,:], thetas)) * self.X_train[i,:] - C * thetas

        return new_thetas


    def update(self, thetas, alpha=0.01, C=1):
        """ C: regularization parameter
        """
        new_thetas = np.zeros((self.d))
        for i in range(self.N):
            for j in range(thetas.size):
                new_thetas[j] = thetas[j] - alpha * (sigmoid(self.X_train[i, :], thetas) - self.Y_train[i]) * self.X_train[i, j] - C * thetas[j]

        return new_thetas


    def converged(self, thetas, new_thetas):
        s = 0
        for i in range(len(thetas)):
            s += abs(thetas[i] - new_thetas[i])
        
        if (1.0 * s) / len(thetas) < 0.000001:
            return True
        return False


    def train(self, alpha=0.1):
        thetas = np.zeros(self.d) # initialization
        new_thetas = np.zeros(self.d) + 0.01 # initialization
        # if you initialize with zeroes, then the regularization parameter will be ignored
        i = 0
        while i < 100 and not self.converged(thetas, new_thetas):
            print "thetas before update: " + str(thetas)
            thetas = new_thetas
            new_thetas = self.update(thetas)
            #new_thetas = self.update_matrix(thetas)
            print "thetas after update: " + str(new_thetas)
            i += 1
        return new_thetas


if __name__ == '__main__':
    X_train = np.array([[1, 2, 3], 
                        [2, 4, 8], 
                        [2, 4, 8], 
                        [2, 2, 2], 
                        [0, 4, 9], 
                        [2, 5, 9]])
    Y_train = np.array([[1],
                        [1], 
                        [1], 
                        [0], 
                        [1], 
                        [1]])

    LR = LogisticRegression(X_train, Y_train)
    LR.train()

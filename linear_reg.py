import numpy as np

class LinearRegression:
    def __init__(self, **options):
        self.options = {
            'learning_rate': 0.01,
            'iterations': 1000,
            **options
        }
        
    def gradient_descent(self, X1, y):
        diff = X1.dot(self.weights) - y
        self.slopes = (X1.transpose().dot(diff) / X1.shape[0]).transpose()
        self.weights = self.weights - self.slopes * self.options['learning_rate']
        
        
    def train(self, X, y):
        X1 = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.zeros(X.shape[1] + 1)
        
        for _ in range(self.options['iterations']):
            self.gradient_descent(X1, y)
            
    
    def test(self, X):
        X1 = np.c_[np.ones(X.shape[0]), X]
        return X1.dot(self.weights)
    
    
    def accuracy(self, X, y):
        X1 = np.c_[np.ones(X.shape[0]), X]
        ss_res = (y - X1.dot(self.weights)).sum()
        ss_total = (y - y.mean()).sum()
        return 1 - (ss_total / ss_res)
        
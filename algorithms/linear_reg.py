import numpy as np

class LinearRegression:
    def __init__(self, **options):
        self.options = {
            'learning_rate': 0.1,
            'iterations': 100,
            'batch_size': 0,
            **options
        }
        
    def __grad_d(self, X1, y):
        diff = X1.dot(self.weights) - y
        self.slopes = X1.transpose().dot(diff) / X1.shape[0]
        self.weights = self.weights - self.slopes * self.options['learning_rate']
        
    def train(self, X, y):
        X1 = np.c_[np.ones(X.shape[0]), X]
        if len(y.shape) == 1:
            self.weights = np.zeros(X1.shape[1])
        elif len(y.shape) == 2:
            self.weights = np.zeros((X1.shape[1], y.shape[1]))
        
        for i in range(self.options['iterations']):
            batch_size = self.options['batch_size'] or X.shape[0]
            batch_qty = X.shape[0] // batch_size
            for j in range(batch_qty):
                    start = batch_size * j
                    end = start + batch_size
                    self.__grad_d(X1[start:end], y[start:end])
            
    def predict(self, X):
        X1 = np.c_[np.ones(X.shape[0]), X]
        return X1.dot(self.weights)
    
    def test(self, X, y):
        ss_res = (y - self.predict(X)).sum()
        ss_total = (y - y.mean()).sum()
        return 1 - (ss_total / ss_res)
    
import numpy as np

class Knn:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.k = int(np.sqrt(X.shape[0]))
            
    def __knn(self, x):
        distance = np.sqrt((self.X - x) ** 2).sum(axis=1)
        xy = np.c_[distance, self.y]
        knn = xy[xy[:, 0].argsort()][:self.k, 1:]
        unique, counts = np.unique(knn, axis=0, return_counts=True)
        index = counts.argmax()
        return unique[index]
    
    def predict(self, X):
        y_pred = np.array([self.__knn(row) for row in X])
        if y_pred.shape[1] == 1:
            return y_pred.reshape(X.shape[0])
        return y_pred
    
    def test(self, X, y):
        correct = (self.predict(X) == y).sum(axis=0)
        return correct / X.shape[0]
    
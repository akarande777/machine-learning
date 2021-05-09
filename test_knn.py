import pandas as pd

dataset = pd.read_csv('data/Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)

# Testing algorithm
from algorithms.knn import Knn
knn = Knn(X_train_sc, y_train)

print(knn.test(X_test_sc, y_test))

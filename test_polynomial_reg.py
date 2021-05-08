import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)

pf = PolynomialFeatures(degree = 4)
X_train_poly = pf.fit_transform(X_train_sc)
X_test_poly = pf.fit_transform(X_test_sc)

# Linear Regression
from algorithms.linear_reg import LinearRegression

le = LinearRegression()
le.train(X_train_poly, y_train)

print(y_train)
print(le.predict(X_train_poly))

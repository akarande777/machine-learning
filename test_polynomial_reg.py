import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Position_Salaries.csv')

X_train = dataset.iloc[:, 1:2].values
y_train = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)

pf = PolynomialFeatures(degree = 4)
X_train_poly = pf.fit_transform(X_train_sc)

# Linear Regression
from algorithms.linear_reg import LinearRegression
reg = LinearRegression()

reg.train(X_train_poly, y_train)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train_poly), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

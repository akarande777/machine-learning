import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Salary_Data.csv')

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

# Linear Regression
from algorithms.linear_reg import LinearRegression

reg = LinearRegression(batch_size=4, iterations=20)
reg.train(X_train_sc, y_train)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train_sc), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

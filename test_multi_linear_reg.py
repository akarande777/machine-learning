import pandas as pd
import numpy as np

dataset = pd.read_csv('data/50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
# Dropping one dummy variable to protect from the dummy variable trap
X = np.array(ct.fit_transform(X))[:, 1:]

# Splitting into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = np.array(X_train)
X_train_sc[:, -3:] = sc.fit_transform(X_train[:, -3:])
X_test_sc = np.array(X_test)
X_test_sc[:, -3:] = sc.fit_transform(X_test[:, -3:])

# Linear Regression
from algorithms.linear_reg import LinearRegression

reg = LinearRegression(batch_size=15, iterations=20)
reg.train(X_train_sc, y_train)

print(y_test)
print(reg.predict(X_test_sc))

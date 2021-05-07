import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/cars.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, :1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# Dropping one dummy variable to protect from the dummy variable trap
y = np.array(ct.fit_transform(y))[:, 1]

# Splitting into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)

# Logistic Regression
from algorithms.logistic_reg import LogisticRegression

le = LogisticRegression(batch_size=15, iterations=20)
le.train(X_train_sc, y_train)

print(le.test(X_test_sc, y_test))

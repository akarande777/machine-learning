import pandas as pd
import numpy as np

dataset = pd.read_csv('data/Cars_Data.csv')

X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 1].values

categorize = lambda x: 'Low' if x < 15 else ('High' if x > 30 else 'Medium')
    
y = np.array([categorize(x) for x in y]).reshape(y.shape[0], 1)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# Dropping one dummy variable to protect from the dummy variable trap
y = np.array(ct.fit_transform(y))[:, :-1]

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

reg = LogisticRegression(batch_size=15, iterations=20)
reg.train(X_train_sc, y_train)

print(reg.test(X_test_sc, y_test))

import pandas as pd
import matplotlib.pyplot as plt
from linear_reg import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data/Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)

le = LinearRegression(learning_rate=0.01)

le.train(X_train_sc, y_train)

size = X_train.shape[0]

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, le.test(X_train_sc).reshape(size, 1), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

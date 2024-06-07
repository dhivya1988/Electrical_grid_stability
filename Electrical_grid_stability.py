import numpy as np
import pandas as pd
df = pd.read_csv('Data_for_UCI_named.csv')
df.head()
df.tail()
x = df.iloc[:, :12].values
print(x)
y = df.iloc[:, -2].values
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.3)
print(x_test)
print(x_train)
print(y_train)
print(y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(x_test)
print(y_pred)
import math
import sklearn.metrics
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print(mse)
mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
print(mae)
rmse = math.sqrt(mse)
print(rmse)
import matplotlib.pyplot as plt
plt.plot(y_test)
plt.plot(y_pred)
plt.plot(y_test)
plt.plot(y_pred)
plt.hist(y_pred)
plt.hist(y_test)
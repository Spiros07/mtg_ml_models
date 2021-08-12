#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('mtg_final_cleaned_no_dupl.csv')
data.head()
data.dtypes


# %%
data.dropna(axis=0, subset=['market price ($)'], inplace=True)
data.dropna(axis=0, subset=['foil price ($)'], inplace=True)

X = data['market price ($)']
y = data['foil price ($)']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)   
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=20)


# print(data)

# print('x = ', X_train.shape)
# print('y = ', y_train.shape)


# %%
reg = linear_model.LinearRegression()

X_train_matrix = X_train.values.reshape(-1,1)
X_val_matrix = X_val.values.reshape(-1,1)
X_test_matrix = X_test.values.reshape(-1,1)


#train the model
reg.fit(X_train_matrix,y_train)

#predictions for val and test set
y_val_pred = reg.predict(X_val_matrix)
y_test_pred = reg.predict(X_test_matrix)

#the coefficients
coef = reg.coef_

#mse
val_mse = round(mean_squared_error(y_val, y_val_pred), 3)
test_mse = round(mean_squared_error(y_test, y_test_pred), 3)


# The coefficient of determination: 1 is perfect prediction
val_r2 = round(r2_score(y_val, y_val_pred), 3)
test_r2 = round(r2_score(y_test, y_test_pred), 3)


intercept = round(reg.intercept_, 4)

print('Coefficient: ', reg.coef_)
print('intercept: ', intercept)
print('val mse: ', val_mse)
print('val_r2: ', val_r2)
print('test mse: ', test_mse)
print('test r2: ', test_r2)


# %%
plt.scatter(X_val_matrix,y_val,marker='1', color='black', label='val set')
plt.scatter(X_test_matrix,y_test,marker='.', color='green', label='test set')
plt.plot(X_val_matrix,y_val_pred, lw=4, c='blue', label='val regression line')
plt.plot(X_test_matrix,y_test_pred, lw=2, c='red', label='test regression line')
plt.xlabel('market price ($)', fontsize=10)
plt.ylabel('foil price ($)', fontsize=10)
plt.legend(loc='best')
plt.show()


# %%

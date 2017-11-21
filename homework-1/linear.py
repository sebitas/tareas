import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from LinearR import LinearReg
import pdb

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# instantiate
linreg = LinearRegression()
mylinreg = LinearReg()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
mylinreg.fit(X_train, y_train)


y_pred = linreg.predict(X_test)
my_y_pred = linreg.predict(X_test)

print(y_pred)
print(my_y_pred)


## Excercise 1 
# Try your own linear regression algorithm 

if ((y_pred == my_y_pred).all()):
	print ('equal')
else:
	print ('not equal')


## Excercise 2, try nonlinear features and measure the RMSE

mylinreg.calculateRMSE(X_train, y_train)

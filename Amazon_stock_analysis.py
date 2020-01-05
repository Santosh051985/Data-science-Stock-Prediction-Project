import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split


amazon = quandl.get("WIKI/AMZN")
print(amazon.tail())
amazon = amazon[['Adj. Close']]
forecast_out = int(30)             # predicting 30 days into future
amazon['Prediction'] = amazon[['Adj. Close']].shift(-forecast_out) 
print(amazon.head())

X = np.array(amazon.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out]          # remove last 30 from X

y = np.array(amazon['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("""       #########       Confidence     #########      """)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(""" ***********Fore cast  Prediction      ***********""")
print(forecast_prediction)

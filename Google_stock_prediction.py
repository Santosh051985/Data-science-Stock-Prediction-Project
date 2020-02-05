import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

# Load data from google
google = quandl.get("WIKI/GOOG")
total_rows = len(google.index)
print(total_rows)
total_col= len(google.columns)
print(total_col)
google.shape
print(google.tail())
google = google[['Adj. Close']]
forecast_out = int(60)             # predicting 60 days into future
google['Prediction'] = google[['Adj. Close']].shift(-forecast_out) 
print("""#####       Print Data    ######""")
print(google.head())

X = np.array(google.drop(['Prediction'], 1))
X = preprocessing.scale(X)
#  set X_forecast equal to last 30
X_forecast = X[-forecast_out:]
# remove last 60 from X

X = X[:-forecast_out]                 
y = np.array(google['Prediction'])
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
#plot Hitogram for fore cast of predicted value of google stock market
%matplotlib inline
import matplotlib.pyplot as plt
google.hist(bins=20,figsize=(20,15))
plt.show()

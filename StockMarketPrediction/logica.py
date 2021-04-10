import math
import pandas as pd
import datetime
import math
import numpy as np
import pandas_datareader.data as web
import pickle
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import l1_min_c

######DISCLAIMER! THIS VIEW IS HORRIFIC, WATCH WITH YOUR OWN RISK OF BLEEDING EYES AND TURNING EYES BACK!, YOU'VE BEEN WARNED!#######

def task1(df):

	try:
		#Get all closing values
		close_px = df['Adj Close']

		#Create moving avergae values
		mavg = close_px.rolling(window=100).mean()

		#Calculate rets
		rets = close_px / close_px.shift(1) - 1

		#Provide data to Flask app
		return close_px.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')), close_px.to_list(), mavg.to_list(), rets.to_list()

	#If any error, provide back to flask app, although it does not work properly.
	except TypeError as e:
		return e
	except NameError as e:
		return e
	except Exception as e:
		return e
	except RemoteDataError as e:
		return e

def task2(data):

	df = data

	dfreg = df.loc[:,['Adj Close','Volume']]
	dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
	dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

	# Drop missing value
	dfreg.fillna(value=-99999, inplace=True)
	# We want to separate 1 percent of the data to forecast
	forecast_out = int(math.ceil(0.01 * len(dfreg)))
	# Separating the label here, we want to predict the AdjClose
	forecast_col = 'Adj Close'
	dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
	X = np.array(dfreg.drop(['label'], 1))
	# Scale the X so that everyone can have the same distribution for linear regression
	X = preprocessing.scale(X)
	# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]
	# Separate label and identify it as y
	y = np.array(dfreg['label'])
	y = y[:-forecast_out]
	
	#Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	##################
	##################
	##################


	# Linear regression
	clfreg = LinearRegression(n_jobs=-1)
	# 1 - First save the models to local device in models folder
	# filename = 'models/clfreg_model.sav'
	# pickle.dump(clfreg, open(filename, 'wb'))

	# 2 - load the models from disk onces first instruction is done once.
	# clfreg = pickle.load(open(filename, 'rb'))
	clfreg.fit(X_train, y_train)


	# KNN Regression
	clfknn = KNeighborsRegressor(n_neighbors=2)
	# Save model to a pickle
	# filename3 = 'models/clfknn_model.sav'
	# pickle.dump(clfknn, open(filename3, 'wb'))
	
	# 2 - load the models from disk onces first instruction is done once.
	# clfknn = pickle.load(open(filename3, 'rb'))
	clfknn.fit(X_train, y_train)


	##################
	##################


	#Create confindence scores
	confidencereg = clfreg.score(X_test, y_test)

	confidenceknn = clfknn.score(X_test, y_test)
	

	# results
	print('The linear regression confidence is:',confidencereg*100)

	print('The knn regression confidence is:',confidenceknn*100)
	

	#Create new columns
	forecast_reg = clfreg.predict(X_lately)

	forecast_knn = clfknn.predict(X_lately)
	

	#Process all new columns data
	dfreg['Forecast_reg'] = np.nan

	last_date = dfreg.iloc[-1].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)

	for i in forecast_reg:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns))]
	    dfreg['Forecast_reg'].loc[next_date] = i
	    
	
	dfreg['Forecast_knn'] = np.nan

	last_date = dfreg.iloc[-26].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	    
	for i in forecast_knn:
	    next_date = next_unix
	    next_unix += datetime.timedelta(days=1)
	    dfreg['Forecast_knn'].loc[next_date] = i
	        
	

	return dfreg.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')), dfreg['Adj Close'].to_list(), dfreg['Forecast_reg'].to_list(), dfreg['Forecast_knn'].to_list()









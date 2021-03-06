4 # -*- coding: utf-8 -*-
"""Copy of Stock_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZFUiHHRJBAeAY8lhioMT5nm6yOstMZru
"""

#To predict the closing stock price for past 60 days for appl.inc

#importing libraries

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers  import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-01-01')
df

df.shape

#To visualize the closing price 
plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=19)
plt.show()

#creating a new dataframe with the close column only 
data = df.filter(['Close'])
#Converting the dataframe to numpy array
dataset = data.values
#Geting the number of rows to train the model 
training_data_len = math.ceil( len(dataset) * .8 )
training_data_len

#Scale the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

#Creation of training dataset 
training_data = scaled_data[0:training_data_len]
#Spliting the dataset into x_train & y_train datasets
x_train = []
y_train = []

for i in range(60, len(training_data)):
  x_train.append(training_data[i-60:i, 0])
  y_train.append(training_data[i, 0]) 

  if i<= 61:
    print(x_train) 
    print(y_train)
    print()

#convert the x_train and y_train into np array
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Building the LSTM model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#FOR Training the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Creating the testing dataset
#Create a new array containing scaled values from index 1348 to 2005
test_data = scaled_data[training_data_len - 60: , :]
#Creating the data x_test and y_test 
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (60, len(test_data)): 
 x_test.append(test_data[i-60:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_test.shape

#Predict the price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Root mean square error
rmse = np.sqrt( np.mean(predictions - y_test )**2 )
rmse

#PLOT THE DATA 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions 
#visualize the data 
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val','Predictions'], loc='lower right ')
plt.show()

valid

#Get the data qoute 
 apple_qoute = web.DataReader('AAPL',data_source='yahoo', start ='2012-01-01', end='2018-12-02')
 #creating new dataframe
 new_df = apple_qoute.filter(['Close'])
 #get the last 60 days value and then convert into numpy array
 last_60_days = new_df[-60:].values
 #scale the  data between 0 and 1
 last_60_days_scaled = scaler.transform(last_60_days)
 #create and empty list 
 X_test = []
#APPEND THE past 60 days 
X_test.append(last_60_days_scaled)
#Convert the X_test data set to numpy array
X_test = np.array(X_test)
#Reshape the data 

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Geting the predicted price 
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

apple_qoute2 = web.DataReader('AAPL',data_source='yahoo', start ='2012-01-01', end='2018-01-02')
print(apple_qoute2['Close'])


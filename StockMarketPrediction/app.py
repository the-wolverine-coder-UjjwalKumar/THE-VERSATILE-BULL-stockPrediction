from flask import Flask,render_template,request,jsonify
from random import sample
import pandas as pd
import gatherer
import company
import logica

#for working of model
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers  import Dense, LSTM

app = Flask(__name__)

symbol = ""
start = ""
end = ""
#data = pd.DataFrame()
comp_name = ""

@app.route('/')
def index():
	return render_template('main.html')

@app.route('/data',methods=["POST", 'GET'])
def data():

	global symbol
	global start
	global end
	global data
	global comp_name

	if request.method=='POST':
		print(request)
		# if symbol != request.form['search']:
		symbol = request.form['search']
		source = request.form['sourcery']
		start = request.form['trip-start']
		end = request.form['trip-end']

		data = gatherer.data(symbol, source, start, end)
		comp_name = company.get_symbol(symbol)

		#print(symbol,comp_name)
		return chart1()

@app.route('/chart1')
def chart1():

	global start
	global end
	global data
	global comp_name

	dt, dd, mav, rets = logica.task1(data)
	print(type(dd))
	return render_template('chart1.html', stock_date=dt, stock_data=dd, mav=mav, company=comp_name, start=start, end=end)

@app.route('/chart2')
def chart2():

	global start
	global end
	global data
	global comp_name

	dt, dd, mav, rets = logica.task1(data)
	return render_template('chart2.html', stock_date=dt, rets=rets, company=comp_name, start=start, end=end)

@app.route('/chart3',methods=["POST", 'GET'])
def chart3():

	global start
	global end
	global data
	global comp_name
	global symbol
	global pred_Price

	if request.method=='POST':
		print(request)
		# if symbol != request.form['search']:
		symbol = request.form['search']
		source = request.form['sourcery']
		start = request.form['trip-start']
		end = request.form['trip-end']

		comp_name = company.get_symbol(symbol)

	
	dt, dd, reg, knn = logica.task2(data)
	#inital setup for model
	df = web.DataReader(name=symbol, data_source='yahoo', start=start, end=end)
	#creating a new dataframe with the close column only 
	data = df.filter(['Close'])
	#Converting the dataframe to numpy array
	dataset = data.values
	#Geting the number of 80 percent of data as traning len 
	training_data_len = math.ceil( len(dataset) * .8 )

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(dataset)

	training_data = scaled_data[0:training_data_len]

	x_train = []
	y_train = []

	for i in range(60, len(training_data)):
		x_train.append(training_data[i-60:i, 0])
		y_train.append(training_data[i, 0]) 

	x_train, y_train = np.array(x_train), np.array(y_train)

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	#Building the LSTM model 
	model = Sequential()
	model.add(LSTM(100, return_sequences=True, input_shape = (x_train.shape[1],1)))
	model.add(LSTM(100, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')

	model.fit(x_train, y_train, batch_size=1, epochs=1)

	#predicting the closing price value for APPL comapny for 30 March 2021
	apple_qoute = web.DataReader(name=symbol,data_source='yahoo', start =start, end=end)
	
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
	print("The predicted price for 30th March 2021 :: "+str(pred_price))

	return render_template('chart3.html', stock_date=dt, stock_data=dd, reg=reg, knn=knn, company=comp_name, start=start, end=end, pred_Price = pred_price)

@app.route("/aboutus")
def aboutus():
	return render_template("aboutus.html")

@app.route("/contact")
def contact():
	return render_template("contact.html")

@app.route("/blog")
def blog():
	return render_template("blog.html")


	

if __name__ == '__main__':
	app.run(debug=1)





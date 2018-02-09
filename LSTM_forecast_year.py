from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
 
number_weeks = 20
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
 
# load dataset
series = read_csv('InputFileName.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(series)
series = numpy.log(series)
#print(series)

# transform data to be stationary
raw_values = series.values
#print(raw_values)
diff_values = difference(raw_values, 1)
#print(diff_values)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
#print(supervised_values)
 
# split data into train and test-s[0:]
train = supervised_values[:]
#print(train)
 
# repeat experiment
repeats = 1
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train, 1, 20, 7)
	print(lstm_model)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train[:, 0].reshape(len(train), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(number_weeks):
		X = train[i, 0:-1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat,number_weeks+1-i)
		# store forecast
		predictions.append(yhat)
		print('Week=%d, Predicted=%f' % (i+1, numpy.exp(yhat)))
'''
Recurrent Neural Network

Installing Theano
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

Installing Tensorflow
pip install tensorflow

Installing Keras
pip install --upgrade keras
'''

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# File / Package Import
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

import numpy
from matplotlib import pyplot
import pandas
from sklearn.preprocessing import MinMaxScaler

#---------------------------------------------------------------------------------------------#
# keras imports
#---------------------------------------------------------------------------------------------#

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Feature Engineering
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# Train
#---------------------------------------------------------------------------------------------#

# Importing the training set
dataset_train = pandas.read_csv('Google_Stock_Price_Train.csv')

# take only hte 'Open" column, index 1
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling; scale values between 0, 1
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# use previous 60 open prices to predict the 61 open price
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# convert to arrays from lists
X_train, y_train = numpy.array(X_train), numpy.array(y_train)

# Reshaping; 3 dimensional array
# dim[0] -> # of records / batch size; in this case the number of 60 days of data
# dim[1] -> # of columns / time stamps; 60 which is the number of previous open prices
# dim[2] -> # of features / input dimension; in this case it is only 'Open' prices; could add volume, close price,
#           another stock; if take open, close, volume this dimension = 3
X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#---------------------------------------------------------------------------------------------#
# Test
#---------------------------------------------------------------------------------------------#

# Getting the real stock price of 2017
dataset_test = pandas.read_csv('Google_Stock_Price_Test.csv')

# take only hte 'Open" column, index 1
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# Feature Scaling; need training dataset for the prevoius 60 time stamps of open prices
dataset_total = pandas.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)

# Feature Scaling; scale values between 0, 1
inputs = sc.transform(inputs)

# Creating a data structure with 60 timesteps and 1 output
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    
# convert to arrays from lists
X_test = numpy.array(X_test)

# Reshaping; 3 dimensional array
# dim[0] -> # of records / batch size; in this case the number of 60 days of data
# dim[1] -> # of columns / time stamps; 60 which is the number of previous open prices
# dim[2] -> # of features / input dimension; in this case it is only 'Open' prices; could add volume, close price,
#           another stock; if take open, close, volume this dimension = 3
X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Build the RNN
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# return_sequences = True because adding LSTM after this layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
# return_sequences = True because adding LSTM after this layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
# return_sequences = True because adding LSTM after this layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# do not need to return the sequences becauset next layer is the output layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer; this is the open price at timestamp 61
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Train the RNN
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
#regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

regressor.save('google_rnn.h5')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Make prediction
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# load model if not in memory
regressor = load_model('google_rnn.h5')

# predict on test set
predicted_stock_price = regressor.predict(X_test)

# convert normal values to stock prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Plot Results
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

pyplot.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
pyplot.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
pyplot.title('Google Stock Price Prediction')
pyplot.xlabel('Time')
pyplot.ylabel('Google Stock Price')
pyplot.legend()
pyplot.show()


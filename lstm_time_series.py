__author__ = 'Debjit'
"""
This class uses Long Short Term Memory Neural Network to solve a time-series problem (Regression Problem) 
"""

import pandas
import math
from math import sqrt
import numpy
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

numpy.random.seed(7)

def trainLSTM(train_X,train_Y,test_X,test_Y,look_back):
                   '''
              @Input : Training and test data  
              @Input_type : Numpy array 
              @Output : trained model
              @Output_type : sequential model
                   '''
                   
                   for i in range(len(train_X)):
                         print(train_X[i].shape, train_Y[i].shape, test_X[i].shape, test_Y[i].shape)
                         train_X[i] = numpy.reshape(train_X[i], (train_X[i].shape[0],1, train_X[i].shape[1]))
                         test_X[i] = numpy.reshape(test_X[i], (test_X[i].shape[0],1, test_X[i].shape[1]))
                         model = Sequential()
                         model.add(LSTM(128, input_shape=(1,look_back), activation='tanh', use_bias=True))
                         model.add(Dense(1, activation="sigmoid"))
                         model.compile(optimizer='adam', loss='mean_squared_error')
                         history = model.fit(train_X[i], train_Y[i], epochs=1, batch_size=5, verbose=2, shuffle=False)
                   return model

# making a prediction and plotting the results
def predict(test_X,test_Y,model):
                        '''
              @Input : test data (Numpy) and Predicting (model) the results  
              @Input_type : Numpy and sequential model
              @Output : output results
              @Output_type : plots 
                         '''

                         columns = ['Open', 'High', 'Low', 'Last', 'Settle', 'Volume', 'Previous Day Open Interest']
                         yhat = model.predict(test_X[i])
                         plt.plot(test_Y[i])
                         plt.plot(yhat)
                         plt.xlabel("Size of test data")
                         plt.ylabel("Normalized data points")
                         plt.legend(['predictions', 'test'], loc='upper left')
                         plt.title("Plotting test against predictions for " + columns[i])
                         #plt.show()
                         transpose_yhat=list(map(list, zip(*yhat)))
                         if i == 0:
                              pred_yhat = transpose_yhat
                         else:
                              pred_yhat = pred_yhat + transpose_yhat
                         print(len(pred_yhat))

# printing the final RMSE and also the test and predicted results
    
                         rmse = sqrt(mean_squared_error(test_Y[i], yhat))
                         print('Test RMSE: %.3f' % rmse)
                         
# reshaping the data for tensor operations and then defining and fitting a LSTM model

                         pred_yhat=[]
                         transpose_test_Y = list(map(list, zip(*test_Y)))
                         print(scaler.inverse_transform(transpose_test_Y))

                         print('*******************************************************')
                         transpose_pred_yhat=list(map(list, zip(*pred_yhat)))
                         print(scaler.inverse_transform(transpose_pred_yhat))

def data_preprocessing(path,look_back):
           '''
              @Input : Data path and look_back_parameter
              @Input_type : str and int
              @Output : Training and test data 
              @Output_type : Numpy array
           '''
   
           dataframe = pandas.read_csv(path, usecols = [1,2,3,4,6,7,8], engine = 'python', skipfooter = 3) ##coloumns you want to chose
           dataframe = dataframe.iloc[::-1].reset_index(drop=True)
           dataset = dataframe.values
           dataset = dataset.astype('float32')

# normalizing the dataset

           scaler = MinMaxScaler(feature_range=(0, 1))
           dataset = scaler.fit_transform(dataset)

# splitting into train and test sets and obtaining their transpose matrices

           train_size = int(len(dataset) * 0.70)
           test_size = len(dataset) - train_size
           train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
           transpose_train = list(map(list, zip(*train)))
           transpose_test = list(map(list, zip(*test)))
           #look_back = 3
           test_X=[]
           test_Y=[]
           train_X=[]
           train_Y=[]

# creating a function that defines the X and Y using a variable look_back window

           for i in range(len(transpose_train)):
                     c,d = create_dataset(transpose_train[i], look_back)
                     train_X.append(c)
                     train_Y.append(d)
           for i in range(len(transpose_test)):
                     c,d = create_dataset(transpose_test[i], look_back)
                     test_X.append(c)
                     test_Y.append(d)

           return train_X, train_Y, test_X, test_Y

def _create_dataset(dataset, look_back):
        
	setX, setY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		setX.append(a)
		setY.append(dataset[i + look_back])
	return numpy.array(setX), numpy.array(setY)


#-------------------------------------------------------------------------------------------------------------------
def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path',default='/Data/Lumber-futures.csv',help="data in csv format")
  parser.add_argument('--look_back',default=3,help='how many rows you want to look back')
  args = parser.parse_args()
  
  train_X, train_Y, test_X, test_Y = data_preprocessing(args.data_path,args.look_back)
  model = trainLSTM(train_X, train_Y, test_X, test_Y,args.look_back)
  predict(test_X,test_Y,model)

if __name__ == '__main__':
    main()






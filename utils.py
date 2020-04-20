import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Masking
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

def preprocessing(df,cols):
  # This function replaces the missing data with NaN. 
  # The data also needs to be converted from string to float
  df = df.replace("?",np.nan)
  df = df[cols].astype('float32')
  print ("Number of missing data points: ")
  print (df.isnull().sum())
  return df

def load_data(data, n_prev = 15):
  # Builds the datset, using 15 lags for the input.  
  docX, docY = [], []
  for i in range(len(data)-n_prev):
    docX.append(data.iloc[i:i+n_prev].values)
    docY.append(data.iloc[i+n_prev].values)
  alsX = np.array(docX)
  alsY = np.array(docY)
  return alsX, alsY

def train_test_split(df, test_size=0.1):  
  #Splits data to training and testing parts
  ntrn = round(len(df) * (1 - test_size))
  X_train, y_train = load_data(df.iloc[0:ntrn])
  X_test, y_test = load_data(df.iloc[ntrn:])
  return (X_train, y_train), (X_test, y_test)

def find_missing_indices(df): 
  #Finds the missing indices and creates the data-set (lag data-set) for predicting the missing values
  missing_value_indices = df[df.isnull().any(axis=1)].index
  X,y = load_data(df.fillna(-1))
  X_missing = X[missing_value_indices]
  return X_missing, missing_value_indices

def fit_model(X_train,y_train):
  # Models an LSTM using keras, with 4 hidden layers, 2 fully connected layers. 
  # 10% of the data is used to validate.
  # mse is used to evaluate the data
  # Adam optimizer is used
  n_timesteps, n_features, n_outputs = X_train.shape[1],X_train.shape[2], y_train.shape[1]
  model = Sequential()
  model.add(Masking(mask_value=-1, input_shape=(n_timesteps, n_features)))
  model.add(LSTM(4, activation='relu', input_shape=(n_timesteps, n_features)))
  model.add(Dense(3, activation='relu'))
  model.add(Dense(n_outputs))
  model.compile(loss='mse', optimizer='adam')
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
  model.fit(X_train, y_train,validation_split = 0.1, epochs=5,callbacks=[es])
  return model

def predict_model(model,X_test):
  # predictions from the model
  return model.predict(X_test)

def evaluate(y_pred,y_test):
  # evaluated using MSE
  mse = mean_squared_error(y_pred,y_test)
  print('Test MSE: %.3f' % mse)
  return mse

def filled_dataset(df,missing_data,indices,cols):
  #imputing the original dataset with the predicted data
  df_missing = pd.DataFrame(data=missing_data, index=indices, columns=cols)
  df_missing[df_missing < 0] = 0
  impute_df = pd.concat([df.dropna(),df_missing])
  return impute_df.sort_index()



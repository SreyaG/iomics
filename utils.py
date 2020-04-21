"""
The helper functions used to impute data using LSTM predictions. 
"""

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
  '''
  This function replaces the missing data with NaN. The data also needs to be converted from string to float.
  
  Parameters
  ----------
  df: Type:Pandas DataFrame
      Dataframe that contains the timeseries raw data. 
  cols: Type: list
       Features of interest.
  
  Returns
  ---------
  df:Type: Pandas DataFrame 
     Timeseries Feature only DataFrame with float values, and missing values replaced with NaN. 
  '''
  df = df.replace("?",np.nan)
  df = df[cols].astype('float32')
  print ("Number of missing data points: ")
  print (df.isnull().sum())
  return df

def load_data(data, n_prev = 15):
  '''
  Builds the datset, using 15 lags for the input.
  Parameters
  ---------- 
  data: Type : DataFrame
       Timeseries dataFrame with features; 
  n_prev: Type: int
       Lag in the data. 
       
  Returns:
  --------
  alsX: Type : numpy array(3D)
        Tensor consisting of data upto "t-n_prev". 
  alsY: Type: Numpy array(2D).
        Target vector at time "t" . 
  '''
  docX, docY = [], []
  for i in range(len(data)-n_prev):
    docX.append(data.iloc[i:i+n_prev].values)
    docY.append(data.iloc[i+n_prev].values)
  alsX = np.array(docX)
  alsY = np.array(docY)
  return alsX, alsY

def train_test_split(df, test_size=0.1):
  '''
  Splits data to training and testing parts
  
  Parameters
  ----------
  df: Type : DataFrame
      Preprocessed time-series dataframe with only the features. Type : DataFrame
  test_size: Type: int
      Fraction of the data used for testing. 
  
  Returns
  --------
  X_train: type: numpy array(3D)
      Data to train the model. This is time series data so its not randomly sampled but kept in contiguous.
  y_train: Type: Numpy array(2D)
     Target vector for training. 
  X_test : Type: Numpy array(3D)
     Fraction of the data used for testing (prediction).
  y_test : Type: Numpy array(2D)
     Target vector for the testing data-set used to evaulate the model.
  ''' 
  ntrn = round(len(df) * (1 - test_size))
  X_train, y_train = load_data(df.iloc[0:ntrn])
  X_test, y_test = load_data(df.iloc[ntrn:])
  return (X_train, y_train), (X_test, y_test)

def find_missing_indices(df): 
  ''' 
  Finds the missing indices and creates the data-set (lag data-set) for predicting the missing values
  Parameters
  -----------
  df : Type: DataFrame
     Preprocessed time-series dataframe with missing data as NaN.
  Returns:
  -----------
  X_missing: Type : Numpy array(3D)
      Data for prediction of missing values. 
  missing_value_indices : Type: list
      Indices of the missing values.
  '''
  missing_value_indices = df[df.isnull().any(axis=1)].index
  X,y = load_data(df.fillna(-1))
  X_missing = X[missing_value_indices]
  return X_missing, missing_value_indices

def evaluate(y_pred,y_test):
  '''
  Evaluate the model using MSE as the metric.
  Parameters:
  ----------
  y_pred : Type: numpy array(2D)
     Predicted data. 
  y_test : Type: numpy array(2D)
     Target data. 
  
  Returns:
  -------
  mse : Type: float
    Mean Squared Error between the prediction and the true values.
  '''
  mse = mean_squared_error(y_pred,y_test)
  print('Test MSE: %.3f' % mse)
  return mse

def filled_dataset(df,missing_data,indices,cols):
  '''
  Imputing the original dataset with the predicted data
  
  Parameters
  ----------
  df : Type: DataFrame
     The complete timeseries dataframe with the missing data. 
  missing_data : Type: numpy array (2D)
     The predicted missing data.
  indices : Type: list
     Indices of the missing data. 
  cols: Type: list
     List of features.
  '''
  df_missing = pd.DataFrame(data=missing_data, index=indices, columns=cols)
  df_missing[df_missing < 0] = 0
  impute_df = pd.concat([df.dropna(),df_missing])
  return impute_df.sort_index()

class lstm_imputer:
  '''
  Models an LSTM using keras, with 4 hidden layers, 2 fully connected layers. 
  10% of the data is used to validate.
  MSE is used to evaluate the data. An adam optimizer is used to converge faster.
  Functions
  -------------------
   __init__ : 
   Intialize the LSTM described above .
   Parameters
   -----------
   n_timesteps: Type: int
     The number of timesteps in the lag.
   n_features : Type:int
     The number of features.
   n_outputs : Type: int
     The number of outputs, size of the target vector. 
     
   fit :
   Train the given LSTM model to the training set.
   
   Parameters
   -----------
   X_train: type: numpy array(3D)
      Data to train the model. This is time series data so its not randomly sampled but kept in contiguous.
   y_train: Type: Numpy array(2D)
     Target vector for training.
   
   predict:
   Use the model to make prediction for a input dataset.
   
   Parameters
   -----------
   X_test : Type: Numpy array(3D)
     Input data used for prediction.
   
   Returns
   ---------
   y_pred : Type: Numpy array(2D)
     Predicted target vector.
   
  ''' 
  def __init__(self,n_timesteps, n_features, n_outputs):
    self.model = Sequential()
    self.model.add(Masking(mask_value=-1, input_shape=(n_timesteps, n_features)))
    self.model.add(LSTM(4, activation='relu', input_shape=(n_timesteps, n_features)))
    self.model.add(Dense(32, activation='relu'))
    self.model.add(Dense(n_outputs))
    self.model.compile(loss='mse', optimizer='adam')
    self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

  def fit(self,X_train,y_train):
    self.model.fit(X_train, y_train,validation_split = 0.1, epochs=5,callbacks=[self.es])
  
  def predict(self,X_test):
    y_pred = self.model.predict(X_test)
    return y_pred

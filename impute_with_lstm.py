
"""

WHAT: 
This code imputes missing data using LSTM. 
The data-set is preprocessed, scaled and partitioned into training and testing data. The missing data points are identified.
The LSTM model is trained with masking to account for missing data.
The model is then used to predict the missing data points.

WHY: 
Time series datasets can have missing data.
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of 
learning order dependence in sequence prediction problems. 
https://www.bioinf.jku.at/publications/older/2604.pdf

An LSTM network can be used to impute missing values . It is better than traditional methods like 
1. interpolation, or 2. imputing with a constant (0 or mean) because the LSTM can predict the missing data based on past 
information.
For non-timeseries data generative adversarial networks (GANs) are shown to work better than traditional 
imputation techniques. 

This code imputes and returns the complete dataset. 

ASSUMES:
1. The input data is sequential and has uniform timesteps. 
2. There is enough data to train LSTM. 
3. There is missing data in the data-set. 

The python libray requirements file is included. 

FUTURE IMPROVEMENTS: 

1. The data-set is large and was truncated to 500,000 data points for training on a local computer. 
Optimization of the data loading can improve the efficiency of the code, and allow training with more 
data improving the model (typical of NN).

2. The choice of parameters in the model was searched over a small grid. Extensive hyper-parameter search will 
help to find the best hyper-parameters for the given model and data-set. 

VARIABLES: 
df : Type: DataFrame
    Dataframe with the entire raw data-set.
df_model : Type: DataFrame
    Pandas dataframe with the entire data-set with the selected features only.
utils : Library of custom functions to preprocess, train and predict from the LSTM model.
    Utility functions included. 
scaler: Function from scikitlearn. 
    Function to scale, and descale the features. The features are scaled between [0,1] to ensure the a smoother gradient
    descent and improved convergence, hence an improved model.
X_train: type: numpy array(3D)
    Data to train the model. This is time series data so its not randomly sampled but kept in contiguous.
y_train: Type: Numpy array(2D)
    Target vector for training. 
X_test : Type: Numpy array(3D)
    Fraction of the data used for testing (prediction).
y_test : Type: Numpy array(2D)
    Target vector for the testing data-set used to evaulate the model.
X_missing: Type : Numpy array(3D)
    Data for prediction of missing values. 
missing_value_indices : Type: list
    Indices of the missing values.
y_pred : Type: numpy array(2D)
    Predicted data. 
y_test : Type: numpy array(2D)
    Target data.
mse : Type: float
    Mean Squared Error between the prediction and the true values.
y_missing : Type: numpy array (2D)
    Predicted values of missing data.
df_fin : Type : DataFrame
    Imputed dataset.

Note: The MSE calculated for the test dataset occurs after re-scaling of the features and is 
larger than the validation/training data. It's not necessarily an effect of over-fitting.

SG 2020/04/20

OUTPUT : Saves the imputed data file as .csv in the working directory.
"""


import argparse
import pandas as pd
import sys
import os
import utils
from sklearn.preprocessing import MinMaxScaler

def main():
  # read in arguments
  parser = argparse.ArgumentParser(description='File Path and/or Features')
  # filepath argument
  parser.add_argument('path',metavar='path',type=str,help='the path to the dataset')
  # optional argument features to use.
  parser.add_argument('--features',metavar='features', type=str, help='Features to impute')
  args = parser.parse_args()
  
  #path to the .csv data-set
  input_path = args.path
  try:
    df = pd.read_csv(input_path,sep=";")
  except:
    sys.exit('Path does not exist')
  print("The first 5 rows of the data-set")
  print(df.head())
  
  #features to use 
  if args.features:
    cols = args.features.split(",")
  else:
    cols=df.columns.tolist()[2:]
  print(cols)
  for feature in cols:
    if feature not in df.columns.tolist():
      print('These features are not present')
      sys.exit()
  
  df = utils.preprocessing(df,cols)
  df_model = df.copy()
  
  #scale the data-set
  scaler = MinMaxScaler()
  df_model[cols] = scaler.fit_transform(df_model[cols].values)
  
  #data preparation
  (X_train, y_train), (X_test, y_test) = utils.train_test_split(df_model.fillna(-1).head(500000))
  X_missing,missing_indices = utils.find_missing_indices(df_model.head())
  
  #fitting the LSTM
  n_timesteps, n_features, n_outputs = X_train.shape[1],X_train.shape[2], y_train.shape[1]
  model = utils.lstm_imputer(n_timesteps, n_features,n_outputs)
  model.fit(X_train,y_train)
  
  #Test set
  y_pred = scaler.inverse_transform(model.predict(X_test))
  mse = utils.evaluate(y_pred,scaler.inverse_transform(y_test))
  
  #Filling the missing data
  y_missing = scaler.inverse_transform(model.predict(X_missing))
  df_fin=utils.filled_dataset(df,y_missing,missing_indices,cols)
  print("Imputed missing data")
  print(df_fin.iloc[missing_indices].head())
  
  #Saving the imputed data-set
  df_fin.to_csv("impute_household_power_consumption.csv")

if __name__ == '__main__':
  main()

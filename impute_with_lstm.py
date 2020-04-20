
"""

WHAT: 
This code imputes missing data using LSTM. 
The data-set is cleaned and partitioned into training and testing data. 
The model is trained using an LSTM with masking to account for missing data.
The model is used to predict the missing data points.

WHY: 
Time series data sets can have missing data. An LSTM can be used to impute missing values with a prediction. 
It is better than traditional methods like 1. interpolation, or 2. imputing with a constant (0 or mean) because
the LSTM can 
This code imputes and returns the complete data.  

ASSUMES: 
The requirements file is included. 

FUTURE IMPROVEMENTS: 

The data-set is large and was truncated to 500,000 data points for training on a local computer. 
Optimization of the data loading can improve the efficiency of the code, and allow training with more data improving the model (typical of NN).

The choice of parameters in the model was searched over a small grid. Extensive hyper-parameter search. 


VARIABLES: 
df : Pandas dataframe with the entire data-set
df_model : Pandas dataframe with the entire data-set with the features only
utils : Utility functions included 
scaler: Function to scale, and descale the features to improve modelling
model : Keras based LSTM model with masking. 4 hidden layers and 2 fully connected layers
10% of the data is used for cross-validation
df_fin : Imputed Data
Note: The MSE calculated for the test dataset occurs after re-scaling of the features and is larger than the validation/training data.
It's not necessarily an effect of over-fitting.

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
  parser = argparse.ArgumentParser(description='File Path and/or Features')
  parser.add_argument('path',metavar='path',type=str,help='the path to the dataset')
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
  model = utils.fit_model(X_train,y_train)
  
  #Test set
  y_pred = scaler.inverse_transform(utils.predict_model(model,X_test))
  mse = utils.evaluate(y_pred,scaler.inverse_transform(y_test))
  
  #Filling the missing data
  y_missing = scaler.inverse_transform(utils.predict_model(model,X_missing))
  df_fin=utils.filled_dataset(df,y_missing,missing_indices,cols)
  print("Imputed missing data")
  print(df_fin.iloc[missing_indices].head())
  
  df_fin.to_csv("impute_household_power_consumption.csv")

if __name__ == '__main__':
  main()

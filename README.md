# Iomics Take Home 
## Usage
Requirements file is included. 

Input : python3 impute_with_lstm.py filepath --features 'feature_1,feature_2'

Output : "impute_household_power_consumption.csv" saved in the working directory with the imputed data points. 

The imputed data-set has values of the datatype float32 instead of string in the original dataset. 

## Task Assigned
1. Identify a well-regarded LSTM implementation and adapt it for re-usability.
2. Download and prepare the data for use from http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption.
3. Using the LSTM, synthesize the missing measurement data.
4. Your focus should be on code quality, documentation, and re-usability. An accurate model is nice to have, but not a requirement.

## Data
The data-set has 2075259 entries covering every minute for 47 months. 
* All calendar timestamps are present in the dataset.
* There are 25979 (~1.25%) rows with missing data, represented by "?".
* There are 7 features 
 * Global_active_power
 * Global_reactive_power
 * Voltage
 * Global_intensity
 * Sub_metering_1
 * Sub_metering_2
 * Sub_metering_3

## Model
The Keras implementation of LSTM was used to model the data.  
https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
This implementation has the ability to use GPUs if available, without additional code. I used Google collaboratory for my implementation which provides free cloud GPU hence allowing me to iterate quickly.
The disadvantage of using Keras implementations is lack of customization.

The model is trained on the first 404,986 rows including missing data. 
Masking is used to ensure the model does not train on the missing data. 

The number of timesteps was determined using hyperparameter search and set to 15.
The features were normalized to lie between [0,1]
Early stopping was used.

## Evaluation

The MSE on the validation set during training is 0.03.



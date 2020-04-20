"""

WHAT: [Purpose of code]

WHY: [Need that the code fulfills]

ASSUMES: [Files needed, libraries required]

FUTURE IMPROVEMENTS: [Optimizations, scaling issues, concerns]

VARIABLES: [Name and description of complex variables expected]

SG 2020/04/20



OUTPUT: The input dataset with missing values filled.

"""



# Import statements

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
import utils as utils

def SampleFunc(var1, var2):

    """

    class/function definitions

    [Short description of function]

    [var1 - description]

    [var2 - description]

    [returns - description]

    """

    return



# Test Code. Will run when this script is executed

if __name__ == '__main__':
  main()

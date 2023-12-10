# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:23:04 2023

@author: Group 11
"""


# DATA PREPROCESSING TEMPLATE

# TO IMPORT LIBRARIES
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TO IMPORT THE DATA SET
dataset = pd.read_csv("Data.csv")

# TO CREATE THE MATRIX OF INDEPENDENT VARIABLE, X
X = dataset.iloc[:, :3].values

# TO CREATE THE MATRIX OF DEPENDENT VARIABLE, Y
Y = dataset.iloc[:, -1:].values

# TO HANDLE MISSING DATA

# A. To know how much of the data is missing
missing_data = dataset.isnull().sum().sort_values(ascending=False)

# B. To impute a value for the missing value
from sklearn.impute import SimpleImputer

simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
simple_imputer = simple_imputer.fit(X[:, 1:3])
X[:, 1:3] = simple_imputer.transform(X[:, 1:3])

''' Other Function
X[:, 1:3] = simple_imputer.fit_transform(X, fit_params=)
'''
# TO ENCODE CATEGORICAL DATA

# A. To encode the Categorical Data (country) in the Independent Variable, X
# Ordinal data is with ranking
# Nominal data is without ranking


# Ordinal
'''
from sklearn.preprocessing import LabelEncoder

label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
'''


# B. To Create the Dummy variable for the categorical Data (Country) in the Independent Variable, X
# Nominal
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

column_transformer = ColumnTransformer(
    [("Country", OneHotEncoder(categories="auto"), [0])],
    remainder="passthrough")

X = column_transformer.fit_transform(X)
X = X.astype(float)

# C. To Encode the Categorical Variable (Purchase) in the Independent Variable, Y

from sklearn.preprocessing import LabelEncoder
label_encoder_Y = LabelEncoder()
Y[:, -1] = label_encoder_Y.fit_transform(Y[:, -1])

# TO SPLIT THE WHOLE DATASET INTO TRAINING DATA SET AND TESTING DATASET

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# TO PERFORM FEATURE SCALING

# A. For Standardization Feature Scaling

from sklearn.preprocessing import StandardScaler # For the data that is NOT normally distributed

standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_train_standard[:, 3:5] = standard_scaler.fit_transform(X_train_standard[:, 3:5])

X_test_standard = X_test.copy()
X_test_standard[:, 3:5] = standard_scaler.transform(X_test_standard[:, 3:5])

# B. For Normalization Feature Scaling

from sklearn.preprocessing import MinMaxScaler # For the data that is normally distributed

normal_scaler = MinMaxScaler()

X_train_normal = X_train.copy()
X_train_normal[:, 3:5] = normal_scaler.fit_transform(X_train_normal[:, 3:5])

X_test_normal = X_test.copy()
X_test_normal[:, 3:5] = normal_scaler.fit_transform(X_test_normal[:, 3:5])

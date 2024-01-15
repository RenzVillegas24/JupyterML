import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump # to serialize
from joblib import load # to deserialize

## To Load the Model
logreg_from_joblib = load('Logistic_regression.pkl')


# To Import the Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")


# To create the matrix of the independent variable
X = dataset.iloc[:, 1:3].values

# to create the matrix of the dependent variable
Y = dataset.iloc[:, 3].values

# To enter the input
X_input = [[65, 845000]]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2,random_state=0)
standard_scaler = StandardScaler()
X_train_standard = X_train.copy()
X_train_standard = standard_scaler.fit_transform(X_train_standard)

X_input_standard = standard_scaler.transform(X_input)

Y_output = logreg_from_joblib.predict(X_input_standard)


# %% Importing the libraries
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%  Setup the dataset

# Read the dataset
dataset = pd.read_csv("movie_metadata.csv")
# Get Rows and Columns size
R_sz, C_sz = dataset.shape
# Get the Independent Variable, X and Dependent Variable, Y
X, Y = dataset.iloc[:,  [x for x in range(C_sz) if x != 9]].copy(), dataset.iloc[:, 9].copy()

# %% Handle missing data

# List of nominal columns
nominal = X.select_dtypes(include='O')
nominal_keys = nominal.keys().tolist()

# List of ordinal columns
ordinal = X.select_dtypes(exclude='O')
ordinal_keys = ordinal.keys().tolist()

# Ordinal data excluded (cannot be averaged)
ordinal_non_averaged = [
    'aspect_ratio',
    'title_year',
    'duration',
    'facenumber_in_poster']

# Data to be excluded
excluded_data = [
    'movie_imdb_link']


# %% Impute the missing data
# Removes the rows with missing data (all the nominals and ordinals that are not averaged)
dataset_copy = dataset.copy()

dataset_copy.dropna(subset=[*ordinal_non_averaged,*nominal_keys], inplace=True)
X, Y = dataset_copy.iloc[:,  [x for x in range(C_sz) if x != 9]].copy(), dataset_copy.iloc[:, 9].copy()

X.drop(columns=excluded_data, inplace=True)
nominal_keys = [x for x in nominal_keys if x not in excluded_data]

# Impute the ordinal data
simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Replace the ordinal data with the imputed data
X[ordinal_keys] = simple_imputer.fit(X[ordinal_keys]).transform(X[ordinal_keys])

# Impute the Y data (gross)
Y = pd.DataFrame(
    simple_imputer.fit(Y.values.reshape(-1, 1)).transform(Y.values.reshape(-1, 1)), 
    columns=['color'])

print(f"{R_sz - dataset_copy.shape[0]} out of {R_sz} rows removed due to missing data, {(R_sz - dataset_copy.shape[0])/R_sz*100:.2f}% of the data")

# %% Dummy variable for the categorical Data
nominal = X.select_dtypes(include='O')
# exclude the columns in excluded_data
nominal = nominal[[x for x in nominal_keys if x not in excluded_data]]

X_encode = OneHotEncoder(sparse_output=False, handle_unknown='error')
X_dummy = pd.DataFrame(
    X_encode.fit_transform(nominal), 
    columns=X_encode.fit(nominal).get_feature_names_out())
X_dummy['color'] = LabelEncoder().fit_transform(X['color'])

# %% Add the ordinal data
X_dummy[ordinal_keys] = X[ordinal_keys]
X.drop(columns=nominal_keys, inplace=True)
X_dummy[X.columns] = X
X = X_dummy

# X.actor_2_name.value_counts().sort_values(ascending=False)

# %% Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)


# %% Feature Scaling
standard_scaler = StandardScaler()


X_train_standard = X_train.copy()
X_train_standard[ordinal_keys] = standard_scaler.fit_transform(X_train_standard[ordinal_keys])

X_test_standard = X_test.copy()
X_test_standard[ordinal_keys] = standard_scaler.transform(X_test_standard[ordinal_keys])

normal_scaler = MinMaxScaler()

X_train_normal = X_train.copy()
X_train_normal[ordinal_keys] = normal_scaler.fit_transform(X_train_normal[ordinal_keys])

X_test_normal = X_test.copy()
X_test_normal[ordinal_keys] = normal_scaler.transform(X_test_normal[ordinal_keys])

# %% Study the missing data (Run this section for studying the missing data only, not used in the main code implementation)'''
'''

# Get all the missing data
missing_datas = X[dataset.isnull().any(axis=1)]
# Get the missing ordinal data
missing_ordinal_data = missing_datas[ordinal].copy()
# Add a column for total missing data count
missing_ordinal_data['total_missing'] = missing_ordinal_data.isnull().sum(axis=1)
missing_ordinal_data = missing_ordinal_data[missing_ordinal_data.total_missing != 0]

# Get the missing nominal data
missing_nominal_data = missing_datas[nominal].copy()
# Add a column for total missing data count
missing_nominal_data['total_missing'] = missing_nominal_data.isnull().sum(axis=1)
missing_nominal_data = missing_nominal_data[missing_nominal_data.total_missing != 0]

'''
#
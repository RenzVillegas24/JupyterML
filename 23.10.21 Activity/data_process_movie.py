# %% Importing the libraries
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# %%  Setup the dataset

# Read the dataset
dataset = pd.read_csv("movie_metadata.csv")
# Get Rows and Columns size
R_sz, C_sz = dataset.shape
# Get the Independent Variable, X and Dependent Variable, Y
X, Y = dataset.iloc[:,  [x for x in range(C_sz) if x != 9]].copy(), dataset.iloc[:, 9].copy()

# %% Handle missing data

# Ordinal data excluded (cannot be averaged)
ordinal_non_averaged = [
    'aspect_ratio',
    'title_year',
    'duration',
    'facenumber_in_poster']

# Data to be excluded
excluded_data = [
    'movie_imdb_link']

# Combined data
combined_data = [
    ('actor_1_name','actor_2_name','actor_3_name')]

label_data = [
    'color']

# List of nominal columns
nominal = X.select_dtypes(include='O')
nominal_keys = [x for x in nominal.keys().tolist() if x not in combined_data[0]]

# List of ordinal columns
ordinal = X.select_dtypes(exclude='O')
ordinal_keys = ordinal.keys().tolist()


# %% Impute the missing data
dataset_copy = dataset.copy()
# Remove the rows with duplicates
X.drop(columns=excluded_data, inplace=True)
dataset_copy.drop_duplicates(inplace=True)

# Removes the rows with missing data (all the nominals and ordinals that are not averaged)
dataset_copy.dropna(subset=[*ordinal_non_averaged,*nominal_keys], inplace=True)
X, Y = dataset_copy.iloc[:,  [x for x in range(C_sz) if x != 9]].copy(), dataset_copy.iloc[:, 9].copy()

X.reset_index(drop=True, inplace=True) # Reset the index

# %% Combine the actor names
dataset_copy['actor_name'] = [[e for e in row if e==e] for row in dataset_copy[['actor_1_name','actor_2_name','actor_3_name']].values.tolist()] 
excluded_data.append('actor_name')

dataset_copy = dataset_copy[[x for x in dataset_copy if x not in ('actor_1_name','actor_2_name','actor_3_name')]].copy()


# %% Impute the missing data
print(f"Original dataset size: {dataset_copy.shape[0]}")
print(f"Dataset size after removing duplicates: {dataset_copy.shape[0]}")

mlb = MultiLabelBinarizer(sparse_output=True)
dataset_copy = dataset_copy.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(dataset_copy.pop('actor_name')),
        index=dataset_copy.index,
        columns=[f"{x}_actor_name" for x in mlb.classes_.tolist()]))

nominal_keys = [x for x in nominal_keys if x not in excluded_data]

# Impute the ordinal data
simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Replace the ordinal data with the imputed data
X[ordinal_keys] = simple_imputer.fit(X[ordinal_keys].values).transform(X[ordinal_keys].values)

# Impute the Y data (gross)
Y = pd.DataFrame(
    simple_imputer.fit(Y.values.reshape(-1, 1)).transform(Y.values.reshape(-1, 1)), 
    columns=['gross'])

print(f"{R_sz - dataset_copy.shape[0]} out of {R_sz} rows removed due to missing data ({(R_sz - dataset_copy.shape[0])/R_sz*100:.2f}% of the data)\nRemoved {excluded_data} column/s")

# %% Dummy variable for the categorical Data
nominal = X.select_dtypes(include='O')
# exclude the columns in excluded_data
nominal = nominal[[x for x in nominal_keys if x not in excluded_data + label_data ]]

print(f"Nominal columns: {nominal.keys().tolist() + label_data}")
print(f"Ordinal columns: {ordinal_keys}")

X_encode = OneHotEncoder(sparse_output=False, handle_unknown='error')
X_dummy = pd.DataFrame(
    X_encode.fit_transform(nominal), 
    columns=X_encode.fit(nominal).get_feature_names_out())

for l in label_data:
    X_dummy[l] = LabelEncoder().fit_transform(X[l])

# %% Add the ordinal data
X.drop(columns=nominal_keys, inplace=True)
X.reset_index(drop=True, inplace=True) # Reset the index

X_dummy[ordinal_keys] = X[ordinal_keys]
X_dummy[X.columns] = X
X = X_dummy
#X.dropna(inplace=True)

# X.actor_2_name.value_counts().sort_values(ascending=False)

# %% Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

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

# %%

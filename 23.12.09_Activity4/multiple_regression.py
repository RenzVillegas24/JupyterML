#%% Multiple Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Import the dataset
dataset = pd.read_csv('50_Startups.csv')

# Analysis of the dataset
# %% A. To know if there is any missing data
missing_data = dataset.isnull().sum().sort_values(ascending=False)


# %% B. To check collumn names and total records
dataset_count = dataset.count()


# %% C. TO view the information about the dataset
dataset_info = dataset.info()
print(dataset_info)


# %% D. To view the statistical summary of the dataset
dataset_describe = dataset.describe(include='all')


# %% To create the matrix of the INDEPENDENT VARIABLE, X (YearsExperience)
X = dataset.iloc[:, 0:4].values # All rows, all columns except the last one


# %% To create the matrix of the DEPENDENT VARIABLE, Y (Salary)
Y = dataset.iloc[:, 4:5].values # All rows, only the last column


# %% To view the scatter plot of the dataset
import seaborn as sns
sns.pairplot(dataset)



# %% To determine the Pearson's coefficient of correlation
from scipy.stats import pearsonr



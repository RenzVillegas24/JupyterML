
import pandas as pd
import re
import numpy as np

df = pd.read_csv("complete laptop data0_utf.csv")

def parse_name(text):
    return re.findall(r"\b\w+\b", text.lower())

'''# get all non-numeric columns
for i in df[df.dtypes[df.dtypes == 'object'].index].columns:
    print(i)
    print(np.unique([parse_name(df[i][j])[0] for j in range(df.shape[0]) if type(df[i][j]) == str]))


'''

def split_space(text):
    return re.findall(r"\b\w+\b", text)

# tokenize the data, get all the unique values and store it in a list
brand_name = np.unique([split_space(i.lower())[0] for i in df['name']])

# replace the name of the brand with the brand name
def parse_name(text):
    text = text.lower()

    for brand in brand_name:
        if brand in split_space(text):
            return brand
    return text
    
# get the brand name from the name column
df['Brand'] = df['name'].apply(lambda x: parse_name(x))


c = 0.67 # for static conversion use this

df['Price'] = round(df['Price'].str.replace(',', '').str.lstrip('?').astype(float) * c, 2)


for i in df[df.dtypes[df.dtypes == 'object'].index].columns:
    print(i)
    print(np.unique([split_space(t.lower())[0] for t in df[i] if type(t) == str]))


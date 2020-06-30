# Explain the project file structure
# Importing the libraries
import numpy as np
import pandas as pd
# Pickle is used to serializing and de-serializing a Python object structure/data
import pickle

# copy and paste the data file onto the project folder, read the csv file using pandas, explore data and transform it
df = pd.read_csv('ccppdata1.csv')
print(df.head())
print(df.shape)
# check if there are any null or missing values and remove them
print("Number of NaN values for the column temperature :", df['AT'].isnull().sum())
print("Number of NaN values for the column exhaust_vacuum :", df['V'].isnull().sum())
print("Number of NaN values for the column ambient_pressure :", df['AP'].isnull().sum())
print("Number of NaN values for the column relative_humidity :", df['RH'].isnull().sum())
print("Number of NaN values for the column energy_output :", df['PE'].isnull().sum())
# no null values found
# Use IQR score method to remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
# Use IQR score method to remove outliers
df2 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df2.shape)
# split data into training and testing data sets
from sklearn.model_selection import train_test_split

train, test = train_test_split(df2, test_size=0.2)
# split data into dependent and independent variables for testing and training
X_train = train[['AT', 'V', 'AP', 'RH']]
y_train = train['PE']
X_test = test[['AT', 'V', 'AP', 'RH']]
y_test = test['PE']
print("Size of X Train:\t",X_train.count())
print("Size of X Test:\t",X_test.count())

# Import the linear regression model

from sklearn.linear_model import LinearRegression
lregressor = LinearRegression(fit_intercept=True)

# Fitting model with training data
lregressor.fit(X_train, y_train)

# Saving model to disk by serializing the data objects writing it for using later on
pickle.dump(lregressor, open('lrModel.pkl', 'wb'))

# Loading model to compare the results (by de-serializing and reading it)
lrModel = pickle.load(open('lrModel.pkl', 'rb'))
print(lrModel.predict([[14, 50, 1000, 100]]))
# import decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
dtregressor = DecisionTreeRegressor(criterion="mae")

# Fitting model with training data
dtregressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(dtregressor, open('dtModel.pkl','wb'))

# Loading model to compare the results
dtModel = pickle.load(open('dtModel.pkl','rb'))
print(dtModel.predict([[14, 50, 1000, 100]]))

# Import the linear regression model

from sklearn.ensemble import RandomForestRegressor
rfRegressor = RandomForestRegressor(criterion='mse')

# Fitting model with training data
rfRegressor.fit(X_train, y_train)

# Saving model to disk by serializing the data objects writing it
pickle.dump(rfRegressor, open('rfModel.pkl', 'wb'))

# Loading model to compare the results (by de-serializing and reading it)
rfModel = pickle.load(open('rfModel.pkl', 'rb'))
print(rfModel.predict([[14, 50, 1000, 100]]))
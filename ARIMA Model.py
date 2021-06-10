# Temperature-forecasting-using-ARIMA-model

#importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()
import matplotlib.pyplot as plt
import statsmodels.api as sm 

#importing data
df=pd.read_csv('weatherHistory.csv')
df.head()
df.info()
df.describe()
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'])

#Ploting target variable with datetime
sns.lineplot(x=df['Formatted Date'], y=df['Temperature (C)'])

#Looking for null values and dropping unwanted values
df.isnull().sum()
df.drop(['Precip Type'], axis = 1)
df.drop(['Daily Summary'], axis = 1)

#Impleting ARIMA
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
         
#Implementing ADF Test 
ad_test(df['Temperature (C)'])

#Plotting Autocorelation Graph
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Temperature (C)'])

#Ploting ACF and PACF to find out p,d,q values
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Temperature (C)'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Temperature (C)'],lags=40,ax=ax2)

# p= AR , D = i , q = MA
#p=3 d=0 q=0

from statsmodels.tsa.arima_model import ARIMA

#Fitting dataset in the model
model = ARIMA(df['Temperature (C)'],order = (3,0,0))
model_fit = model.fit()


#Predicting future data
df['Forecast'] = model_fit.predict()
df[['Temperature (C)','Forecast']].plot(figsize=(12,8))

 
from sklearn import metrics

#Result of MAE
print (metrics.mean_absolute_error(df['Temperature (C)'],df['Forecast']))

#Result of MSE
print (metrics.mean_squared_error(df['Temperature (C)'],df['Forecast']))

#Result of RMSE
print (np.sqrt(metrics.mean_squared_error(df['Temperature (C)'],df['Forecast'])))

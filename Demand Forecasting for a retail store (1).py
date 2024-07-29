#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[5]:


df = pd.read_csv('Downloads\\sales.csv')


# In[6]:


print(df.head())
print(df.info())
print(df.describe())


# In[7]:


plt.figure(figsize=(12, 6))
plt.plot(df, label='Sales')
plt.title('Historical Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[ ]:


decomposition = seasonal_decompose(df['sales'], model='additive')
fig = decomposition.plot()
plt.show()


# In[ ]:


train = df.iloc[:-12]
test = df.iloc[-12:]


# In[ ]:


model = ExponentialSmoothing(train['sales'], seasonal='add', seasonal_periods=12).fit()


# In[ ]:


forecast = model.forecast(steps=12)


# In[ ]:


plt.figure(figsize=(12, 6))
plt.plot(train.index, train['sales'], label='Train')
plt.plot(test.index, test['sales'], label='Test')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test['sales'], forecast)
print(f'Mean Absolute Error: {mae}')


# In[ ]:





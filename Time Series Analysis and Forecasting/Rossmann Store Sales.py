#!/usr/bin/env python
# coding: utf-8

# <h1>Sales Data Analysis and Forcasting</h1>
# <P>This project mainly focuse on the <b>time series analysis and forcasting</b>. I have collected 'Rossmann Store Sales' data from <a href="https://www.kaggle.com/c/rossmann-store-sales">Kaggle</a></P>
# 
# <ul>
#     <li><a href="#1" style="text-decoration: none;">Exploratory Data Analysis</a></li>
#     <li><a href="#2" style="text-decoration: none;">ECDF: empirical cumulative distribution function</a></li>
#     <li><a href="#3" style="text-decoration: none;">Missing values</a></li>
#     <li><a href="#4" style="text-decoration: none;">Store types</a></li>
#     <li><a href="#5" style="text-decoration: none;">Correlational Analysis</a></li>
#     <li><a href="#6" style="text-decoration: none;">Conclusion of EDA</a></li>
#     <li><a href="#7" style="text-decoration: none;">Time-Series Analysis per Store Type</a></li>
#     <li><a href="#8" style="text-decoration: none;">Seasonality</a></li>
#     <li><a href="#9" style="text-decoration: none;">Yearly trend</a></li>
#     <li><a href="#10" style="text-decoration: none;">Autocorrelaion</a></li>
#     <li><a href="#11" style="text-decoration: none;">Time Series Analysis and Forecasting with Prophet</a></li>
#     <li><a href="#12" style="text-decoration: none;">Conclusion of Time Series forecasting</a></li>
# </ul>

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pandas import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from statsmodels.distributions.empirical_distribution import ECDF

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from prophet import Prophet


# In[2]:


train = pd.read_csv('Data/train.csv', parse_dates=True, low_memory=False, index_col='Date')
store = pd.read_csv('Data/store.csv', low_memory=False)
train.index


# <h1 id='1'>Exploratory Data Analysis</h1>
# <p>In this section we will explore the missing value of train and store data. Then I will create a new feature for furture analysis</p>

# In[3]:


print('Total :', train.shape)
train.head(5)


# In[ ]:





# <b><i>Sort Description about train data</i></b>
# <ul>
#     <li><i>Store</i>: Indicate the store number</li>
#     <li><i>DayOfWeek</i>: Day number in a week</li>
#     <li><i>Sales</i>: Daily Sales</li>
#     <li><i>Customers</i>: Number of castomer in a day</li>
#     <li><i>Open</i>: The store is open or not. 0 for closed and 1 for Open</li>
#     <li><i>Promo</i>: The store is runing any promotion in that day. 0 for not and 1 for Yes</li>
#     <li><i>StateHoliday</i>: Indicate State Holiday was or not. 0 for not and 1 for Yes</li>
#     <li><i>SchoolHoliday</i>: Indicate School Holiday was or not. 0 for not and 1 for Yes</li>
# </ul>
# <p>Since we are dealing with time series analysis, so it is importatn to note all of the things</p>

# In[4]:


train['Day'] = train.index.day
train['Month'] = train.index.month
train['Year'] = train.index.year
train['Weekofyear'] = train.index.weekofyear
train['SalesPerCustomer'] = train['Sales']/train['Customers']
train.head(2)


# In[5]:


train.info()


# In[6]:


NullCheck = train[train['SalesPerCustomer'].isnull()]
NullCheck.head()


# In[7]:


NullCheck['Customers'].unique()


# In[8]:


NullCheck['Sales'].unique()


# In[9]:


NullCheck = train[train['SalesPerCustomer'] == 0]
NullCheck.head()


# <p>Most of the SalesPerCustomer value is zero, neither did not sales at that day nor none of the customer did not come to buy. We need filter Zero Customer and Sales from the train dataset</p>

# In[10]:


train[(train['Sales'] != 0) & (train['Customers'] != 0)].shape


# In[11]:


train['SalesPerCustomer'].describe()


# <p>As we can see, some of the day did not sales or customer did not come. And maximum Sales per Customer ~65. Since std is ~2.2 we can say most of the value is near ~9.49.</p>
# <h3 id='2'>ECDF: empirical cumulative distribution function</h3>

# In[12]:


plt.figure(figsize=(20,10))
fnum = 311
variable = ['Sales', 'Customers', 'SalesPerCustomer']
for x in variable:
    plt.subplot(fnum)
    plt.subplots_adjust(hspace=0.5)
    cdf = ECDF(train[x])
    plt.plot(cdf.x, cdf.y, label='statmodels')
    plt.xlabel(x); plt.ylabel('ECDF');
    fnum=fnum + 1


# <p>Almost 20% of data was zero sales/ Castomer and 80% of time selas less than 1000. So we have to undesrtand, why 20% are zero sales</p>
# <h3 id='3'>Missing values</h3>
# <p>There're 172817 closed stores in the data. we need to avoid any biased during forecasts, this is why we will drop these  zero values. Let check the others, where the store is open but did not sales.</p>

# In[13]:


train['Open'].unique()


# In[14]:


train[(train['Sales'] == 0) & (train['Open'] == 0)].head()


# In[15]:


not_seal = train[(train['Open'] != 0) & (train['Sales'] == 0)]
not_seal.shape


# There are 54 days did not sales any product, I think there are another resone. Let's continue our analysis.</p>

# In[111]:


train = train[(train['Sales'] != 0) & (train['Customers'] != 0)]
train.shape


# </p><b>Let's analysis the store data</b></p>

# In[17]:


store.head()


# In[18]:


store.shape


# In[19]:


store['Store'].unique()


# <b><i>Sort Description about store data</i></b>
# <ul>
#     <li><i>Store</i>: Have an unique ID for each store</li>
#     <li><i>StoreType</i>: Four different types of store (a, b, c, d)</li>
#     <li><i>Assortment</i>: Lavel: a = Basic, b = Extra, C = Extended</li>
#     <li><i>CompetitionDistance</i>: Distance in meters to identify nearest competiitor. </li>
#     <li><i>CompetitionOpenSinceMonth</i>: Nerest competitor open their store in a month.</li>
#     <li><i>CompetitionOpenSinceYear</i>: Nerest competitor open their store in a year.</li>
#     <li><i>Promo2</i>: 0 for store did not perticiate, 1 for did.</li>
#     <li><i>Promo2SinceWeek</i>: Describe the weekly time to reperesent the store perticipation.</li>
#     <li><i>Promo2SinceYear</i>: Describe the yearly time to reperesent the store perticipation.</li>
#     <li><i>PromoInterval</i>: Promotional round starts in February, May, August, November of any given year for that store</li>
# </ul>

# In[20]:


store.isnull().sum()


# <p>As we can see their are some value are missing. We have to deal with this missing value. Let's start with <i>CompetitionDistance</i> values.</p>

# In[21]:


store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)


# In[22]:


store['Promo2'].unique()


# In[23]:


store[store['CompetitionDistance'].isnull() & (store['Promo2'] == 1)].shape


# <p>If no promotion, that's mean thire are no infomation. we can replace it with 0 value</p>

# In[24]:


store.fillna(0, inplace = True)
store.isnull().sum()


# In[25]:


len(train['Store'].unique())


# In[26]:


len(store['Store'].unique())


# <p><b>Merge the two data set for analysis purpose.</b></p>

# In[27]:


train_store = pd.merge(train, store, how='inner', on='Store')
train_store.shape


# In[28]:


train_store.head()


# <h3 id='4'>Store types</h3>

# In[29]:


train_store.groupby('StoreType')['Sales'].describe()


# <p>In description, we can see most of the sales in storeType <i>a</i>. And less dispersion  of data in <i>d</i> Storetype. On other hand, most everage sales in storeType <i>b</i></p>

# In[30]:


train_store.groupby('StoreType')['Sales', 'Customers'].sum()


# <p>Clearly stores of type A. StoreType D goes on the second place in both Sales and Customers. What about date periods?</p>

# In[31]:


sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
              col = 'StoreType',
              palette = 'plasma',
              hue = 'StoreType',
              row = 'Promo')
plt.show()


# In[32]:


sns.factorplot(data=train_store, x='Month', y = 'Customers',
              col = 'StoreType',
              pattern='plasma',
              hue = 'StoreType',
              row = 'Promo')
plt.show()


# In[33]:


sns.factorplot(data = train_store, x='Month', y = 'SalesPerCustomer',
              col='StoreType',
              palette = 'plasma',
              hue = 'StoreType',
              row = 'Promo2')
plt.show()


# <p>All store types follow the same trend. But pretty different between without promotion and with promotion.</p>

# In[34]:


sns.factorplot(data=train_store, x = 'Month', y = 'Sales',
              col = 'DayOfWeek',
              palette = 'plasma',
              hue = 'StoreType',
              row = 'StoreType')
plt.show()


# <p>Although all of the type follow almost same trend. we can see from visualization, most of the sales do better at the end of the day. And most of the seles in storeType b</p>

# In[35]:


# competition open time (in months)
train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) +         (train_store.Month - train_store.CompetitionOpenSinceMonth)
    
# Promo open time
train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) +         (train_store.Weekofyear - train_store.Promo2SinceWeek) / 4.0


# In[36]:


train_store.loc[train_store['Promo2SinceYear'] == 0, 'PromoOpen'] = 0
pd.set_option('display.max_columns', None)
train_store.head()


# In[37]:


train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()


# <p>The most selling and crowded StoreType A doesn't appear to be the one the most exposed to competitors. Instead it's a StoreType B, which also has the longest running period of promotion.</p>

# In[1]:


<h3 id='5'>Correlational Analysis</h3>


# In[38]:


corr_all = train_store.drop('Open', axis=1).corr()
plt.figure(figsize=(20, 20))
mask = np.zeros_like(corr_all, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_all, cmap='mako', mask = mask, annot = True, linewidths= .2, square=True)
plt.show()


# <p>The heatmap is showing that a strong correlation between sales and customer.</p>

# In[39]:


sns.factorplot(data=train_store, x = 'DayOfWeek', y = 'Sales',
              col='Promo',
              row = 'Promo2',
              hue = 'Promo2')
plt.show()


# <p>In case of no promotion, Sales tend to peak on Sunday. Though we should note that StoreType C doesn't work on Sundays. So it is mainly data from StoreType A, B and D. On the contrary, stores that run the promotion tend to make most of the Sales on Monday. This fact could be a good indicator for Rossmann marketing campaigns. The same trend follow the stores where bosth promotion run in same time.</p>
# <h3 id='6'>Conclusion of EDA</h3>
# <ul>
#     <li>Most sales StoreType is A</li>
#     <li>The best "Sale per Customer" StoreType is D</li>
#     <li>Low "Sale per Customer" amount StoreType is B</li>
#     <li>Customers tends to buy more on Modays when one promotion is running and on Sundays when there's no promotion run.</li>
# </ul>

# <h3 id='7'>Time-Series Analysis per Store Type</h3>
# <p>What does mean of Time-Series?</p>
# <p>Time series analysis, a time series is modeled to determine its components in terms of seasonal patterns, trends, relation to external factors. In other hand, time series forecasting uses the information in a time series to forecast future values of that series.</p>
# <p>I am building a time series analysis on store types instead of individual stores. The main advantage of this approach is its simplicity of presentation and overall account for different trends and seasonalities in the dataset.</p>
# <h3 id='8'>Seasonality</h3>
# <p>Here I will take four stores types to understand the group</p>
# <ul>
#     <li>Store number 2 for <i>StoreType A</i></li>
#     <li>Store number 85 for <i>StoreType B</i></li>
#     <li>Store number 1 for <i>StoreType C</i></li>
#     <li>Store number 14 for <i>StoreType D</i></li>
# <ul>

# In[48]:


# preparation: input should be float type
train['Sales'] = train['Sales'].astype(float)

# store types
sales_a = train[train.Store == 2]['Sales']
sales_b = train[train.Store == 85]['Sales'].sort_index(ascending = True)
sales_c = train[train.Store == 1]['Sales']
sales_d = train[train.Store == 14]['Sales'].sort_index(ascending = True)

f, (axa, axb, axc, axd) = plt.subplots(4, figsize = (12, 13))

# store types
sales_a.resample('W').sum().plot(ax = axa)
sales_b.resample('W').sum().plot(ax = axb)
sales_c.resample('W').sum().plot(ax = axc)
sales_d.resample('W').sum().plot(ax = axd)
plt.show()


# <p>Retail sales for StoreType A and C tend to peak for the Christmas season and then decline after the holidays.</p>
# <h3 id='9'>Yearly trend</h3>
# <p>Retail sales for StoreType A and C tend to peak for the Christmas season and then decline after the holidays.</p>

# In[49]:


from statsmodels.tsa.seasonal import seasonal_decompose
f, (axa, axb, axc, axd) = plt.subplots(4, figsize = (12, 13))

# monthly
decomposition_a = seasonal_decompose(sales_a, model = 'additive', freq = 365)
decomposition_a.trend.plot(ax = axa)

decomposition_b = seasonal_decompose(sales_b, model = 'additive', freq = 365)
decomposition_b.trend.plot(ax = axb)

decomposition_c = seasonal_decompose(sales_c, model='additive', freq = 365)
decomposition_c.trend.plot(ax=axc)

decomposition_d = seasonal_decompose(sales_d, model = 'additive', freq = 365)
decomposition_d.trend.plot(ax = axd)
plt.show()


# <p>Overall sales seems increasing. But in StoreType C is shwoing the sales in decrease sequentially. It could not back to the previous point</p>
# <h3 id='10'>Autocorrelaion</h3>
# <p>The next step in ourtime series analysis is to review Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.</p>
# <p><b>ACF</b> is an (complete) auto-correlation function which gives us values of auto-correlation of any series with its lagged values. In simple terms, it describes how well the present value of the series is related with its past values. On toher the hand,<b>PACF</b> is a partial auto-correlation function. Basically instead of finding correlations of present with lags like ACF, it finds correlation of the residuals with the next lag value hence ‘partial’ and not ‘complete’ as we remove already found variations before we find the next correlation.</p>

# In[63]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
plt.figure(figsize=(12, 8))
plt.subplot(421)
plot_acf(sales_a, lags=50, ax = plt.gca())
plt.subplot(422)
plot_pacf(sales_a, lags=50, ax = plt.gca())

plt.subplot(423)
plot_acf(sales_b, lags=50, ax = plt.gca())
plt.subplot(424)
plot_pacf(sales_b, lags=50, ax = plt.gca())

plt.subplot(425)
plot_acf(sales_c, lags=50, ax=plt.gca())
plt.subplot(426)
plot_pacf(sales_c, lags = 50, ax=plt.gca())

plt.subplot(427)
plot_acf(sales_d, lags=50, ax=plt.gca())
plt.subplot(428)
plot_pacf(sales_d, lags=50, ax=plt.gca())
plt.show()


# <p>Type A and type B: Both types show seasonalities at certain lags. For type A, it is each 12th observation with positives spikes at the 12 (s) and 24(2s) lags and so on. For type B it's a weekly trend with positives spikes at the 7(s), 14(2s), 21(3s) and 28(4s) lags.Additionally, in Type C and type D: Plots of these two types are more complex. It seems like each observation is coorrelated to its adjacent observations.</p>
# <h2 id='11'>Time Series Analysis and Forecasting with Prophet</h2>
# <p>The Core Data Science team at Facebook recently published a new procedure for forecasting time series data called Prophet. We have observed two main themes in the practice of creating a variety of business forecasts:</p>
# <ul>
#     <li>Completely automatic forecasting techniques can be brittle and they are often too inflexible to incorporate useful assumptions or heuristics.</li>
#     <li>Analysts who can produce high quality forecasts are quite rare because forecasting is a specialized data science skill requiring substantial experience.</li>
# <ul>

# In[70]:


data = pd.read_csv('Data/train.csv')
data = data[(data['Open'] != 0) & (data['Open'] != 0)]
sales = data[data.Store == 1].loc[:, ['Date', 'Sales']]
sales = sales.sort_index(ascending = False)
sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales = sales.rename(columns = {'Date': 'ds', 'Sales':'y'})
sales.head()


# In[77]:


sales.set_index('ds').plot(figsize=(12, 4))
plt.title(' Date vs Sales')
plt.xlabel('Date')
plt.ylabel('Number of sales')
plt.show()


# <h3 id='12'>Modeling Holidays</h3>
# <p>Prophet also allows to model for holidays, and that's what we do here.</p>

# In[94]:


stateHoliday = data[(data['StateHoliday'] == 'a') | (data['StateHoliday'] == 'b') | (data['StateHoliday'] == 'c')].loc[:, 'Date'].values
schoolHoliday = data[data['SchoolHoliday'] == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'stateHoliday', 'ds': pd.to_datetime(stateHoliday)})
school = pd.DataFrame({'holiday': 'schoolHoliday', 'ds': pd.to_datetime(schoolHoliday)})
holidays = pd.concat((state, school))
holidays.head()


# In[95]:


model = Prophet(interval_width = 0.95, holidays=holidays)
model.fit(sales)


# In[99]:


future_date = model.make_future_dataframe(periods = 6*7)
future_date.head()


# In[100]:


forecast = model.predict(future_date)
forecast.tail(7)


# In[102]:


model.plot(forecast)
plt.show()


# <p>The black dots plots is the observed values of our time series, blue line represent the forecasted values and the blue shaded regions is for the uncertainty intervals of our forecasts. As we see Prophet catches the trends and most of the time gets future values right.</p>
# 
# <p>One other particularly strong feature of Prophet is its ability to return the components of our forecasts. This can help reveal how daily, weekly and yearly patterns of the time series plus manyally included holidayes contribute to the overall forecasted values:</p>

# In[107]:


model.plot_components(forecast)


# <p>The first plot shows that the monthly sales of store number 1 has been linearly decreasing over time. The second graph shows the holiays gaps. The third plot is for the weekly volume of last week sales peaks towards the Monday of the next week, while the forth plot shows that the most buzy season occurs during the Christmas holidays.</p>

# In[110]:


forecast = forecast[['ds', 'yhat']].rename(columns = {'ds':'Data', 'yhat':'Forecast'})
forecast.head()


# <h3 id='12'>Conclusion of Time Series forecasting</h3>
# <p>In conclusion section, I will present main advantages and drawbacks of time series forecasting:</p>
# <p><i><b>Advantages</b></i></p>
# <ul>
#     <li>A powerful tool for the time series forecasting as it accounts for time dependencies, seasonalities and holidays</li>
# </ul>
# <p><i><b>Drawbacks</b></i></p>
# <ul>
#     <li>Doesn't catch interactions between external features</li>
#     <li>Fitting seasonal ARIMA model needs 4 to 5 whole seasons in the dataset, which can be the biggest drawback for new companies.</li>
# </ul>

# In[ ]:





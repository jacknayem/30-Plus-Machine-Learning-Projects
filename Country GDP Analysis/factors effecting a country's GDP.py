#!/usr/bin/env python
# coding: utf-8

# <h1>Factors effecting a country's GDP</h1>
# <ul>
#     <li><a href="#1" style="text-decoration: none;">Why we are working about?</a></li>
#     <li><a href="#2" style="text-decoration: none;">What data we used?</a></li>
#     <li>
#         <ul>
#             <li><a href="#3" style="text-decoration: none;">Load Python libray and Data</a></li>
#             <li><a href="#4" style="text-decoration: none;">Preprocess and Anlysis</a></li>
#             <li><a href="#5" style="text-decoration: none;">Prediction</a></li>
#             <li><a href="#6" style="text-decoration: none;">Visualization</a></li>
#         </ul>
#     </li>
# </ul>
# 
# <h3 id='1'>Why we are predicting?</h3>
# <p>GDP is important because it show you the country's health. Also it demonstrate the economical position. Even indicate the size of economy. Doing prediction a country's GDP can take step more easily.</p>
# 
# <h3 id='2'>What data is avaiable</h3>
# <p>To analys the GDP, we have used <a href="https://www.kaggle.com/stieranka/predicting-gdp-world-countries/data">Kaggle Dataset</a>. This Data compiled from <a href="https://gsociology.icaap.org/dataupload.html">US Government</a></p>
# 
# <h3 id='3'>Load Python libray and Data</h3>
# <p>After download the data. It time to load python libray.</p>

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv('countries of the world.csv', decimal=',')
data.shape


# <h3 id='4'>Preprocess and Anlysis</h3>
# <p>As we can see, we have 227 instance to fit model and predict. Also we have 20 columns. Our trget columns is <b>GDP ($ per capita)</b></p>

# In[5]:


data.columns


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# <p>Is this data. we have a lots of missing data, this is why we need to preprocess. To eliminate, We have two option to eliminate this problem. One is remove those index where the any instence is null. Another one is eliminate those null applying some technique.</p>

# In[9]:


data.describe(include='all')


# In[10]:


data.groupby('Region')[['GDP ($ per capita)','Literacy (%)','Agriculture']].median()


# In[11]:


for col in data.columns.values:
    if data[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_value = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_value = data.groupby('Region')['Climate'].median()
    for region in data['Region'].unique():
        data[col].loc[(data[col].isnull())& (data['Region'] == region)] = guess_value[region]


# In[12]:


data.isnull().sum()


# In[13]:


fig, ex = plt.subplots(figsize=(16, 6))
top_gdp_countries = data.sort_values('GDP ($ per capita)', ascending=False).head(20)
gdps = pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']]],ignore_index=True)

figGDP = sns.barplot(x='Country', y = 'GDP ($ per capita)', data=gdps)
for p in figGDP.patches:
    figGDP.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va = 'center', xytext=(0,9), textcoords = 'offset points')
plt.xticks(rotation=90)
plt.show()


# <p>Most top 20 country by GDP per capital</p>

# In[14]:


plt.figure(figsize=(16, 12))
sns.heatmap(data[2:].corr(), annot=True, fmt='.2f', cmap='rocket_r')
plt.show()


# <p>Top six correlation between GDP per capital and others input variables<p>

# In[15]:


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
corr_GDP = pd.Series()
for col in data.columns[2:]:
    if ((col != 'GDP ($ per capita)')& (col != 'Climate')):
        corr_GDP[col] = data['GDP ($ per capita)'].corr(data[col])
corr_GDP = corr_GDP.abs().sort_values(ascending = False)
for x in range(2):
    for y in range(3):
        sns.regplot(x=corr_GDP.index[y if x == 0 else y+3], y='GDP ($ per capita)', data=data, ax=axes[x,y], fit_reg=False)
        axes[x,y].set_title('Correlation:'+str(corr_GDP.values[y if x == 0 else y+3]))
plt.show()


# <p>Now it's time prepare for model</p>

# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Region_label'] = le.fit_transform(data['Region'])


# In[17]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.2, shuffle = True)
train_feature = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
       'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'Service', 'Region_label']
target = ['GDP ($ per capita)']
X_train = train[train_feature]
y_train = train[target]
X_test = test[train_feature]
y_test = test[target]


# In[18]:


train.shape, test.shape


# <h3 id='5'>Prediction</h3>

# In[19]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
model_name, train_r2, test_r2, train_SMSE, test_SMSE, train_MAE, test_MAE  = ([] for i in range(7))
prediction = pd.DataFrame()
models = {'SVR':SVR(C=1.0, epsilon=0.2),'Ridge':Ridge(alpha=1.0, normalize=True),'LinearRegression':LinearRegression(),'RandomForestRegressor':RandomForestRegressor(n_estimators = 50,
                             max_depth = 6,
                             min_weight_fraction_leaf = 0.05,
                             max_features = 0.8,
                             random_state = 42)}

for name, model  in models.items():
    model.fit(X_train, y_train.values.ravel())
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    prediction[name+'_pred'] = np.concatenate((train_pred, test_pred))
    model_name.append(name)
    train_r2.append(r2_score(y_train, train_pred))
    test_r2.append(r2_score(y_test, test_pred))
    train_SMSE.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_SMSE.append(np.sqrt(mean_squared_error(y_test, test_pred)))
    train_MAE.append(mean_absolute_error(y_train, train_pred))
    test_MAE.append(mean_absolute_error(y_test, test_pred))


# In[20]:


info = pd.DataFrame({'train_r2':train_r2, 'test_r2':test_r2, 'train_SMSE':train_SMSE,
                 'test_SMSE':test_SMSE, 'train_MAE':train_MAE, 'test_MAE':test_MAE}, index = model_name)
info


# <h3 id='6'>Visualization</h3>

# In[21]:


prediction['Actual'] = data['GDP ($ per capita)']
prediction['Region'] = data['Region']
fig, exes = plt.subplots(figsize=(16,12))
sns.scatterplot(x='RandomForestRegressor_pred', y = 'Actual', hue='Region', data=prediction)
plt.xlabel('Actual GDP ($ per capita)')
plt.ylabel('Predicted GDP ($ per capita)')
plt.show()


# <p><b>Top ten country by Total GDP</b></p>

# In[22]:


data['Total GDP $'] = data['GDP ($ per capita)'] * data['Population']
Top_10_total_GDP = data.sort_values('Total GDP $', ascending=False).head(10)
Top_10_total_GDP = Top_10_total_GDP.reset_index(drop=True)
plt.figure(figsize=(16,12))
sns.barplot(x='Country', y = 'Total GDP $', data=Top_10_total_GDP)
plt.show()


# <p><b>Top ten country by Total GDP and the rank of those country by GDP per capital</b></p>

# In[23]:


data = data.sort_values('GDP ($ per capita)', ascending=False)
for country in Top_10_total_GDP['Country']:
    print("{} : {}".format(country,int(data[data['Country'] == country].index.values)))


# In[ ]:





# In[ ]:





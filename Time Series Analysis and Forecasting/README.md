<h1>Sales Data Analysis and Forcasting</h1>
<P>This project mainly focuse on the <b>time series analysis and forcasting</b>. I have collected 'Rossmann Store Sales' data from <a href="https://www.kaggle.com/c/rossmann-store-sales">Kaggle</a></P>

<ul>
    <li><a href="#1" style="text-decoration: none;">Exploratory Data Analysis</a></li>
    <li><a href="#2" style="text-decoration: none;">ECDF: empirical cumulative distribution function</a></li>
    <li><a href="#3" style="text-decoration: none;">Missing values</a></li>
    <li><a href="#4" style="text-decoration: none;">Store types</a></li>
    <li><a href="#5" style="text-decoration: none;">Correlational Analysis</a></li>
    <li><a href="#6" style="text-decoration: none;">Conclusion of EDA</a></li>
    <li><a href="#7" style="text-decoration: none;">Time-Series Analysis per Store Type</a></li>
    <li><a href="#8" style="text-decoration: none;">Seasonality</a></li>
    <li><a href="#9" style="text-decoration: none;">Yearly trend</a></li>
    <li><a href="#10" style="text-decoration: none;">Autocorrelaion</a></li>
    <li><a href="#11" style="text-decoration: none;">Time Series Analysis and Forecasting with Prophet</a></li>
    <li><a href="#12" style="text-decoration: none;">Modeling Holidays</a></li>
    <li><a href="#13" style="text-decoration: none;">Conclusion of Time Series forecasting</a></li>
</ul>


```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pandas import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from statsmodels.distributions.empirical_distribution import ECDF

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from prophet import Prophet
```


```python
train = pd.read_csv('Data/train.csv', parse_dates=True, low_memory=False, index_col='Date')
store = pd.read_csv('Data/store.csv', low_memory=False)
train.index
```




    DatetimeIndex(['2015-07-31', '2015-07-31', '2015-07-31', '2015-07-31',
                   '2015-07-31', '2015-07-31', '2015-07-31', '2015-07-31',
                   '2015-07-31', '2015-07-31',
                   ...
                   '2013-01-01', '2013-01-01', '2013-01-01', '2013-01-01',
                   '2013-01-01', '2013-01-01', '2013-01-01', '2013-01-01',
                   '2013-01-01', '2013-01-01'],
                  dtype='datetime64[ns]', name='Date', length=1017209, freq=None)



<h1 id='1'>Exploratory Data Analysis</h1>
<p>In this section we will explore the missing value of train and store data. Then I will create a new feature for furture analysis</p>


```python
print('Total :', train.shape)
train.head(5)
```

    Total : (1017209, 8)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>2</td>
      <td>5</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>3</td>
      <td>5</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>4</td>
      <td>5</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>5</td>
      <td>5</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

<b><i>Sort Description about train data</i></b>
<ul>
    <li><i>Store</i>: Indicate the store number</li>
    <li><i>DayOfWeek</i>: Day number in a week</li>
    <li><i>Sales</i>: Daily Sales</li>
    <li><i>Customers</i>: Number of castomer in a day</li>
    <li><i>Open</i>: The store is open or not. 0 for closed and 1 for Open</li>
    <li><i>Promo</i>: The store is runing any promotion in that day. 0 for not and 1 for Yes</li>
    <li><i>StateHoliday</i>: Indicate State Holiday was or not. 0 for not and 1 for Yes</li>
    <li><i>SchoolHoliday</i>: Indicate School Holiday was or not. 0 for not and 1 for Yes</li>
</ul>
<p>Since we are dealing with time series analysis, so it is importatn to note all of the things</p>


```python
train['Day'] = train.index.day
train['Month'] = train.index.month
train['Year'] = train.index.year
train['Weekofyear'] = train.index.weekofyear
train['SalesPerCustomer'] = train['Sales']/train['Customers']
train.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Day</th>
      <th>Month</th>
      <th>Year</th>
      <th>Weekofyear</th>
      <th>SalesPerCustomer</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>9.482883</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>2</td>
      <td>5</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>9.702400</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1017209 entries, 2015-07-31 to 2013-01-01
    Data columns (total 13 columns):
     #   Column            Non-Null Count    Dtype  
    ---  ------            --------------    -----  
     0   Store             1017209 non-null  int64  
     1   DayOfWeek         1017209 non-null  int64  
     2   Sales             1017209 non-null  int64  
     3   Customers         1017209 non-null  int64  
     4   Open              1017209 non-null  int64  
     5   Promo             1017209 non-null  int64  
     6   StateHoliday      1017209 non-null  object 
     7   SchoolHoliday     1017209 non-null  int64  
     8   Day               1017209 non-null  int64  
     9   Month             1017209 non-null  int64  
     10  Year              1017209 non-null  int64  
     11  Weekofyear        1017209 non-null  int64  
     12  SalesPerCustomer  844340 non-null   float64
    dtypes: float64(1), int64(11), object(1)
    memory usage: 108.6+ MB
    


```python
NullCheck = train[train['SalesPerCustomer'].isnull()]
NullCheck.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Day</th>
      <th>Month</th>
      <th>Year</th>
      <th>Weekofyear</th>
      <th>SalesPerCustomer</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>292</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>876</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-30</th>
      <td>292</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-30</th>
      <td>876</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-29</th>
      <td>292</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
NullCheck['Customers'].unique()
```




    array([0], dtype=int64)




```python
NullCheck['Sales'].unique()
```




    array([0], dtype=int64)




```python
NullCheck = train[train['SalesPerCustomer'] == 0]
NullCheck.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Day</th>
      <th>Month</th>
      <th>Year</th>
      <th>Weekofyear</th>
      <th>SalesPerCustomer</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-04-29</th>
      <td>1100</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>4</td>
      <td>2014</td>
      <td>18</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-04-25</th>
      <td>948</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>4</td>
      <td>2013</td>
      <td>17</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<p>Most of the SalesPerCustomer value is zero, neither did not sales at that day nor none of the customer did not come to buy. We need filter Zero Customer and Sales from the train dataset</p>


```python
train[(train['Sales'] != 0) & (train['Customers'] != 0)].shape
```




    (844338, 13)




```python
train['SalesPerCustomer'].describe()
```




    count    844340.000000
    mean          9.493619
    std           2.197494
    min           0.000000
    25%           7.895563
    50%           9.250000
    75%          10.899729
    max          64.957854
    Name: SalesPerCustomer, dtype: float64



<p>As we can see, some of the day did not sales or customer did not come. And maximum Sales per Customer ~65. Since std is ~2.2 we can say most of the value is near ~9.49.</p>
<h3 id='2'>ECDF: empirical cumulative distribution function</h3>


```python
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
```


![png](img/output_17_0.png)


<p>Almost 20% of data was zero sales/ Castomer and 80% of time selas less than 1000. So we have to undesrtand, why 20% are zero sales</p>
<h3 id='3'>Missing values</h3>
<p>There're 172817 closed stores in the data. we need to avoid any biased during forecasts, this is why we will drop these  zero values. Let check the others, where the store is open but did not sales.</p>


```python
train['Open'].unique()
```




    array([1, 0], dtype=int64)




```python
train[(train['Sales'] == 0) & (train['Open'] == 0)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Day</th>
      <th>Month</th>
      <th>Year</th>
      <th>Weekofyear</th>
      <th>SalesPerCustomer</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-07-31</th>
      <td>292</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>876</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-30</th>
      <td>292</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-30</th>
      <td>876</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-07-29</th>
      <td>292</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
not_seal = train[(train['Open'] != 0) & (train['Sales'] == 0)]
not_seal.shape
```




    (54, 13)



There are 54 days did not sales any product, I think there are another resone. Let's continue our analysis.</p>


```python
train = train[(train['Sales'] != 0) & (train['Customers'] != 0)]
train.shape
```




    (844338, 13)



</p><b>Let's analysis the store data</b></p>


```python
store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>a</td>
      <td>a</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>c</td>
      <td>c</td>
      <td>620.0</td>
      <td>9.0</td>
      <td>2009.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>a</td>
      <td>29910.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
store.shape
```




    (1115, 10)




```python
store['Store'].unique()
```




    array([   1,    2,    3, ..., 1113, 1114, 1115], dtype=int64)



<b><i>Sort Description about store data</i></b>
<ul>
    <li><i>Store</i>: Have an unique ID for each store</li>
    <li><i>StoreType</i>: Four different types of store (a, b, c, d)</li>
    <li><i>Assortment</i>: Lavel: a = Basic, b = Extra, C = Extended</li>
    <li><i>CompetitionDistance</i>: Distance in meters to identify nearest competiitor. </li>
    <li><i>CompetitionOpenSinceMonth</i>: Nerest competitor open their store in a month.</li>
    <li><i>CompetitionOpenSinceYear</i>: Nerest competitor open their store in a year.</li>
    <li><i>Promo2</i>: 0 for store did not perticiate, 1 for did.</li>
    <li><i>Promo2SinceWeek</i>: Describe the weekly time to reperesent the store perticipation.</li>
    <li><i>Promo2SinceYear</i>: Describe the yearly time to reperesent the store perticipation.</li>
    <li><i>PromoInterval</i>: Promotional round starts in February, May, August, November of any given year for that store</li>
</ul>


```python
store.isnull().sum()
```




    Store                          0
    StoreType                      0
    Assortment                     0
    CompetitionDistance            3
    CompetitionOpenSinceMonth    354
    CompetitionOpenSinceYear     354
    Promo2                         0
    Promo2SinceWeek              544
    Promo2SinceYear              544
    PromoInterval                544
    dtype: int64



<p>As we can see their are some value are missing. We have to deal with this missing value. Let's start with <i>CompetitionDistance</i> values.</p>


```python
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)
```


```python
store['Promo2'].unique()
```




    array([0, 1], dtype=int64)




```python
store[store['CompetitionDistance'].isnull() & (store['Promo2'] == 1)].shape
```




    (0, 10)



<p>If no promotion, that's mean thire are no infomation. we can replace it with 0 value</p>


```python
store.fillna(0, inplace = True)
store.isnull().sum()
```




    Store                        0
    StoreType                    0
    Assortment                   0
    CompetitionDistance          0
    CompetitionOpenSinceMonth    0
    CompetitionOpenSinceYear     0
    Promo2                       0
    Promo2SinceWeek              0
    Promo2SinceYear              0
    PromoInterval                0
    dtype: int64




```python
len(train['Store'].unique())
```




    1115




```python
len(store['Store'].unique())
```




    1115



<p><b>Merge the two data set for analysis purpose.</b></p>


```python
train_store = pd.merge(train, store, how='inner', on='Store')
train_store.shape
```




    (844338, 22)




```python
train_store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Day</th>
      <th>Month</th>
      <th>...</th>
      <th>SalesPerCustomer</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>...</td>
      <td>9.482883</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5020</td>
      <td>546</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>7</td>
      <td>...</td>
      <td>9.194139</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>4782</td>
      <td>523</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>7</td>
      <td>...</td>
      <td>9.143403</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>5011</td>
      <td>560</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>7</td>
      <td>...</td>
      <td>8.948214</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>6102</td>
      <td>612</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>27</td>
      <td>7</td>
      <td>...</td>
      <td>9.970588</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



<h3 id='4'>Store types</h3>


```python
train_store.groupby('StoreType')['Sales'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>StoreType</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>457042.0</td>
      <td>6925.697986</td>
      <td>3277.351589</td>
      <td>46.0</td>
      <td>4695.25</td>
      <td>6285.0</td>
      <td>8406.00</td>
      <td>41551.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>15560.0</td>
      <td>10233.380141</td>
      <td>5155.729868</td>
      <td>1252.0</td>
      <td>6345.75</td>
      <td>9130.0</td>
      <td>13184.25</td>
      <td>38722.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>112968.0</td>
      <td>6933.126425</td>
      <td>2896.958579</td>
      <td>133.0</td>
      <td>4916.00</td>
      <td>6408.0</td>
      <td>8349.25</td>
      <td>31448.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>258768.0</td>
      <td>6822.300064</td>
      <td>2556.401455</td>
      <td>538.0</td>
      <td>5050.00</td>
      <td>6395.0</td>
      <td>8123.25</td>
      <td>38037.0</td>
    </tr>
  </tbody>
</table>
</div>



<p>In description, we can see most of the sales in storeType <i>a</i>. And less dispersion  of data in <i>d</i> Storetype. On other hand, most everage sales in storeType <i>b</i></p>


```python
train_store.groupby('StoreType')['Sales', 'Customers'].sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
      <th>Customers</th>
    </tr>
    <tr>
      <th>StoreType</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>3165334859</td>
      <td>363541431</td>
    </tr>
    <tr>
      <th>b</th>
      <td>159231395</td>
      <td>31465616</td>
    </tr>
    <tr>
      <th>c</th>
      <td>783221426</td>
      <td>92129705</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1765392943</td>
      <td>156904995</td>
    </tr>
  </tbody>
</table>
</div>



<p>Clearly stores of type A. StoreType D goes on the second place in both Sales and Customers. What about date periods?</p>


```python
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
              col = 'StoreType',
              palette = 'plasma',
              hue = 'StoreType',
              row = 'Promo')
plt.show()
```


![png](img/output_46_0.png)



```python
sns.factorplot(data=train_store, x='Month', y = 'Customers',
              col = 'StoreType',
              pattern='plasma',
              hue = 'StoreType',
              row = 'Promo')
plt.show()
```


![png](img/output_47_0.png)



```python
sns.factorplot(data = train_store, x='Month', y = 'SalesPerCustomer',
              col='StoreType',
              palette = 'plasma',
              hue = 'StoreType',
              row = 'Promo2')
plt.show()
```


![png](img/output_48_0.png)


<p>All store types follow the same trend. But pretty different between without promotion and with promotion.</p>


```python
sns.factorplot(data=train_store, x = 'Month', y = 'Sales',
              col = 'DayOfWeek',
              palette = 'plasma',
              hue = 'StoreType',
              row = 'StoreType')
plt.show()
```


![png](img/output_50_0.png)


<p>Although all of the type follow almost same trend. we can see from visualization, most of the sales do better at the end of the day. And most of the seles in storeType b</p>


```python
# competition open time (in months)
train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
        (train_store.Month - train_store.CompetitionOpenSinceMonth)
    
# Promo open time
train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
        (train_store.Weekofyear - train_store.Promo2SinceWeek) / 4.0
```


```python
train_store.loc[train_store['Promo2SinceYear'] == 0, 'PromoOpen'] = 0
pd.set_option('display.max_columns', None)
train_store.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Day</th>
      <th>Month</th>
      <th>Year</th>
      <th>Weekofyear</th>
      <th>SalesPerCustomer</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>CompetitionOpen</th>
      <th>PromoOpen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>9.482883</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>82.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>5020</td>
      <td>546</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>9.194139</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>82.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>4782</td>
      <td>523</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>9.143403</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>82.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>5011</td>
      <td>560</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>8.948214</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>82.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>6102</td>
      <td>612</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>27</td>
      <td>7</td>
      <td>2015</td>
      <td>31</td>
      <td>9.970588</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>82.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
      <th>Customers</th>
      <th>PromoOpen</th>
      <th>CompetitionOpen</th>
    </tr>
    <tr>
      <th>StoreType</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>6925.697986</td>
      <td>795.422370</td>
      <td>10.958105</td>
      <td>7115.514452</td>
    </tr>
    <tr>
      <th>b</th>
      <td>10233.380141</td>
      <td>2022.211825</td>
      <td>3.717593</td>
      <td>11364.495244</td>
    </tr>
    <tr>
      <th>c</th>
      <td>6933.126425</td>
      <td>815.538073</td>
      <td>13.964386</td>
      <td>6745.418694</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6822.300064</td>
      <td>606.353935</td>
      <td>14.827699</td>
      <td>9028.526526</td>
    </tr>
  </tbody>
</table>
</div>



<p>The most selling and crowded StoreType A doesn't appear to be the one the most exposed to competitors. Instead it's a StoreType B, which also has the longest running period of promotion.</p>


```python
<h3 id='5'>Correlational Analysis</h3>
```


      File "<ipython-input-1-2a277d740d74>", line 1
        <h3 id='5'>Correlational Analysis</h3>
        ^
    SyntaxError: invalid syntax
    



```python
corr_all = train_store.drop('Open', axis=1).corr()
plt.figure(figsize=(20, 20))
mask = np.zeros_like(corr_all, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_all, cmap='mako', mask = mask, annot = True, linewidths= .2, square=True)
plt.show()
```


![png](img/output_57_0.png)


<p>The heatmap is showing that a strong correlation between sales and customer.</p>


```python
sns.factorplot(data=train_store, x = 'DayOfWeek', y = 'Sales',
              col='Promo',
              row = 'Promo2',
              hue = 'Promo2')
plt.show()
```


![png](img/output_59_0.png)


<p>In case of no promotion, Sales tend to peak on Sunday. Though we should note that StoreType C doesn't work on Sundays. So it is mainly data from StoreType A, B and D. On the contrary, stores that run the promotion tend to make most of the Sales on Monday. This fact could be a good indicator for Rossmann marketing campaigns. The same trend follow the stores where bosth promotion run in same time.</p>
<h3 id='6'>Conclusion of EDA</h3>
<ul>
    <li>Most sales StoreType is A</li>
    <li>The best "Sale per Customer" StoreType is D</li>
    <li>Low "Sale per Customer" amount StoreType is B</li>
    <li>Customers tends to buy more on Modays when one promotion is running and on Sundays when there's no promotion run.</li>
</ul>

<h3 id='7'>Time-Series Analysis per Store Type</h3>
<p>What does mean of Time-Series?</p>
<p>Time series analysis, a time series is modeled to determine its components in terms of seasonal patterns, trends, relation to external factors. In other hand, time series forecasting uses the information in a time series to forecast future values of that series.</p>
<p>I am building a time series analysis on store types instead of individual stores. The main advantage of this approach is its simplicity of presentation and overall account for different trends and seasonalities in the dataset.</p>
<h3 id='8'>Seasonality</h3>
<p>Here I will take four stores types to understand the group</p>
<ul>
    <li>Store number 2 for <i>StoreType A</i></li>
    <li>Store number 85 for <i>StoreType B</i></li>
    <li>Store number 1 for <i>StoreType C</i></li>
    <li>Store number 14 for <i>StoreType D</i></li>
<ul>


```python
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
```


![png](img/output_62_0.png)


<p>Retail sales for StoreType A and C tend to peak for the Christmas season and then decline after the holidays.</p>
<h3 id='9'>Yearly trend</h3>
<p>Retail sales for StoreType A and C tend to peak for the Christmas season and then decline after the holidays.</p>


```python
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
```


![png](img/output_64_0.png)


<p>Overall sales seems increasing. But in StoreType C is shwoing the sales in decrease sequentially. It could not back to the previous point</p>
<h3 id='10'>Autocorrelaion</h3>
<p>The next step in ourtime series analysis is to review Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.</p>
<p><b>ACF</b> is an (complete) auto-correlation function which gives us values of auto-correlation of any series with its lagged values. In simple terms, it describes how well the present value of the series is related with its past values. On toher the hand,<b>PACF</b> is a partial auto-correlation function. Basically instead of finding correlations of present with lags like ACF, it finds correlation of the residuals with the next lag value hence ‘partial’ and not ‘complete’ as we remove already found variations before we find the next correlation.</p>


```python
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
```


![png](img/output_66_0.png)


<p>Type A and type B: Both types show seasonalities at certain lags. For type A, it is each 12th observation with positives spikes at the 12 (s) and 24(2s) lags and so on. For type B it's a weekly trend with positives spikes at the 7(s), 14(2s), 21(3s) and 28(4s) lags.Additionally, in Type C and type D: Plots of these two types are more complex. It seems like each observation is coorrelated to its adjacent observations.</p>
<h2 id='11'>Time Series Analysis and Forecasting with Prophet</h2>
<p>The Core Data Science team at Facebook recently published a new procedure for forecasting time series data called Prophet. We have observed two main themes in the practice of creating a variety of business forecasts:</p>
<ul>
    <li>Completely automatic forecasting techniques can be brittle and they are often too inflexible to incorporate useful assumptions or heuristics.</li>
    <li>Analysts who can produce high quality forecasts are quite rare because forecasting is a specialized data science skill requiring substantial experience.</li>
<ul>


```python
data = pd.read_csv('Data/train.csv')
data = data[(data['Open'] != 0) & (data['Open'] != 0)]
sales = data[data.Store == 1].loc[:, ['Date', 'Sales']]
sales = sales.sort_index(ascending = False)
sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales = sales.rename(columns = {'Date': 'ds', 'Sales':'y'})
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1014980</th>
      <td>2013-01-02</td>
      <td>5530</td>
    </tr>
    <tr>
      <th>1013865</th>
      <td>2013-01-03</td>
      <td>4327</td>
    </tr>
    <tr>
      <th>1012750</th>
      <td>2013-01-04</td>
      <td>4486</td>
    </tr>
    <tr>
      <th>1011635</th>
      <td>2013-01-05</td>
      <td>4997</td>
    </tr>
    <tr>
      <th>1009405</th>
      <td>2013-01-07</td>
      <td>7176</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.set_index('ds').plot(figsize=(12, 4))
plt.title(' Date vs Sales')
plt.xlabel('Date')
plt.ylabel('Number of sales')
plt.show()
```


![png](img/output_69_0.png)


<h3 id='12'>Modeling Holidays</h3>
<p>Prophet also allows to model for holidays, and that's what we do here.</p>


```python
stateHoliday = data[(data['StateHoliday'] == 'a') | (data['StateHoliday'] == 'b') | (data['StateHoliday'] == 'c')].loc[:, 'Date'].values
schoolHoliday = data[data['SchoolHoliday'] == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'stateHoliday', 'ds': pd.to_datetime(stateHoliday)})
school = pd.DataFrame({'holiday': 'schoolHoliday', 'ds': pd.to_datetime(schoolHoliday)})
holidays = pd.concat((state, school))
holidays.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holiday</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>stateHoliday</td>
      <td>2015-06-04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>stateHoliday</td>
      <td>2015-06-04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stateHoliday</td>
      <td>2015-06-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stateHoliday</td>
      <td>2015-06-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>stateHoliday</td>
      <td>2015-06-04</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = Prophet(interval_width = 0.95, holidays=holidays)
model.fit(sales)
```

    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    




    <prophet.forecaster.Prophet at 0x4faf817240>




```python
future_date = model.make_future_dataframe(periods = 6*7)
future_date.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
forecast = model.predict(future_date)
forecast.tail(7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>holidays</th>
      <th>holidays_lower</th>
      <th>holidays_upper</th>
      <th>schoolHoliday</th>
      <th>schoolHoliday_lower</th>
      <th>schoolHoliday_upper</th>
      <th>stateHoliday</th>
      <th>stateHoliday_lower</th>
      <th>stateHoliday_upper</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>816</th>
      <td>2015-09-05</td>
      <td>4461.862974</td>
      <td>2456.822108</td>
      <td>5618.390013</td>
      <td>4459.024905</td>
      <td>4464.299203</td>
      <td>-369.508201</td>
      <td>-369.508201</td>
      <td>-369.508201</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>273.053304</td>
      <td>273.053304</td>
      <td>273.053304</td>
      <td>-642.561505</td>
      <td>-642.561505</td>
      <td>-642.561505</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4092.354773</td>
    </tr>
    <tr>
      <th>817</th>
      <td>2015-09-06</td>
      <td>4461.755443</td>
      <td>2418.072694</td>
      <td>5716.743604</td>
      <td>4458.671419</td>
      <td>4464.281483</td>
      <td>-374.484075</td>
      <td>-374.484075</td>
      <td>-374.484075</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>272.720208</td>
      <td>272.720208</td>
      <td>272.720208</td>
      <td>-647.204283</td>
      <td>-647.204283</td>
      <td>-647.204283</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4087.271369</td>
    </tr>
    <tr>
      <th>818</th>
      <td>2015-09-07</td>
      <td>4461.647913</td>
      <td>2482.685526</td>
      <td>5716.055377</td>
      <td>4458.344890</td>
      <td>4464.269375</td>
      <td>-299.532167</td>
      <td>-299.532167</td>
      <td>-299.532167</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>349.869927</td>
      <td>349.869927</td>
      <td>349.869927</td>
      <td>-649.402094</td>
      <td>-649.402094</td>
      <td>-649.402094</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4162.115745</td>
    </tr>
    <tr>
      <th>819</th>
      <td>2015-09-08</td>
      <td>4461.540382</td>
      <td>1981.277766</td>
      <td>5244.382390</td>
      <td>4458.117302</td>
      <td>4464.258569</td>
      <td>-797.136279</td>
      <td>-797.136279</td>
      <td>-797.136279</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-148.492652</td>
      <td>-148.492652</td>
      <td>-148.492652</td>
      <td>-648.643626</td>
      <td>-648.643626</td>
      <td>-648.643626</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3664.404103</td>
    </tr>
    <tr>
      <th>820</th>
      <td>2015-09-09</td>
      <td>4461.432851</td>
      <td>1983.683159</td>
      <td>5166.052789</td>
      <td>4457.832484</td>
      <td>4464.247112</td>
      <td>-910.276390</td>
      <td>-910.276390</td>
      <td>-910.276390</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-265.795557</td>
      <td>-265.795557</td>
      <td>-265.795557</td>
      <td>-644.480833</td>
      <td>-644.480833</td>
      <td>-644.480833</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3551.156461</td>
    </tr>
    <tr>
      <th>821</th>
      <td>2015-09-10</td>
      <td>4461.325321</td>
      <td>1786.436117</td>
      <td>5193.720837</td>
      <td>4457.562717</td>
      <td>4464.253758</td>
      <td>-998.640856</td>
      <td>-998.640856</td>
      <td>-998.640856</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-362.094336</td>
      <td>-362.094336</td>
      <td>-362.094336</td>
      <td>-636.546520</td>
      <td>-636.546520</td>
      <td>-636.546520</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3462.684465</td>
    </tr>
    <tr>
      <th>822</th>
      <td>2015-09-11</td>
      <td>4461.217790</td>
      <td>2137.064226</td>
      <td>5339.167372</td>
      <td>4457.335601</td>
      <td>4464.327510</td>
      <td>-743.830766</td>
      <td>-743.830766</td>
      <td>-743.830766</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-119.260894</td>
      <td>-119.260894</td>
      <td>-119.260894</td>
      <td>-624.569872</td>
      <td>-624.569872</td>
      <td>-624.569872</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3717.387024</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.plot(forecast)
plt.show()
```


![png](img/output_75_0.png)


<p>The black dots plots is the observed values of our time series, blue line represent the forecasted values and the blue shaded regions is for the uncertainty intervals of our forecasts. As we see Prophet catches the trends and most of the time gets future values right.</p>

<p>One other particularly strong feature of Prophet is its ability to return the components of our forecasts. This can help reveal how daily, weekly and yearly patterns of the time series plus manyally included holidayes contribute to the overall forecasted values:</p>


```python
model.plot_components(forecast)
```


![png](img/output_77_0.png)


<p>The first plot shows that the monthly sales of store number 1 has been linearly decreasing over time. The second graph shows the holiays gaps. The third plot is for the weekly volume of last week sales peaks towards the Monday of the next week, while the forth plot shows that the most buzy season occurs during the Christmas holidays.</p>


```python
forecast = forecast[['ds', 'yhat']].rename(columns = {'ds':'Data', 'yhat':'Forecast'})
forecast.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Data</th>
      <th>Forecast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-02</td>
      <td>5480.036913</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-03</td>
      <td>5270.349363</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-04</td>
      <td>5408.989513</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-05</td>
      <td>5707.317953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-07</td>
      <td>5629.713614</td>
    </tr>
  </tbody>
</table>
</div>



<h3 id='13'>Conclusion of Time Series forecasting</h3>
<p>In conclusion section, I will present main advantages and drawbacks of time series forecasting:</p>
<p><i><b>Advantages</b></i></p>
<ul>
    <li>A powerful tool for the time series forecasting as it accounts for time dependencies, seasonalities and holidays</li>
</ul>
<p><i><b>Drawbacks</b></i></p>
<ul>
    <li>Doesn't catch interactions between external features</li>
    <li>Fitting seasonal ARIMA model needs 4 to 5 whole seasons in the dataset, which can be the biggest drawback for new companies.</li>
</ul>


```python

```

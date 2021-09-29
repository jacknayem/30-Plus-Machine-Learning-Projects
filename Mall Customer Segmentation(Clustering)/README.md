<h1>Clustering</h1>
<ul>
    <li><a href="#1" style="text-decoration: none;">Content</a></li>
    <li><a href="#2" style="text-decoration: none;">What data we used?</a></li>
    <li>
        <ul>
            <li><a href="#3" style="text-decoration: none;">Load Python libray and data</a></li>
            <li><a href="#4" style="text-decoration: none;">Data Analysis</a></li>
            <li><a href="#5" style="text-decoration: none;">Fit the cluster model</a></li>
            <li><a href="#6" style="text-decoration: none;">Conclution</a></li>
        </ul>
    </li>
</ul>

<h3 id='1'>Content</h3>
<p>You are owing a supermarket mall and through membership cards , you have some basic data about your customers like Customer ID, age, gender, annual income and spending score.
Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.</p>

<p><b>Problem Statement</b></p>
<p>You own the mall and want to understand the customers like who can be easily converge [Target Customers] so that the sense can be given to marketing team and plan the strategy accordingly.</p>

<h3 id='2'>Is data avaiable?</h3>
<p>Yes, the data is avaiable in <a href="https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python">Kaggle</a></p>

<h3 id='3'>Load Python libray and data</h3>


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")
df = pd.read_csv('data.csv')
```

<h3 id='4'>Data Analysis</h3>


```python
df.head()
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
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (200, 5)




```python
df.describe()
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
      <th>CustomerID</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>100.500000</td>
      <td>38.850000</td>
      <td>60.560000</td>
      <td>50.200000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>57.879185</td>
      <td>13.969007</td>
      <td>26.264721</td>
      <td>25.823522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>50.750000</td>
      <td>28.750000</td>
      <td>41.500000</td>
      <td>34.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.500000</td>
      <td>36.000000</td>
      <td>61.500000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>150.250000</td>
      <td>49.000000</td>
      <td>78.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>200.000000</td>
      <td>70.000000</td>
      <td>137.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 5 columns):
     #   Column                  Non-Null Count  Dtype 
    ---  ------                  --------------  ----- 
     0   CustomerID              200 non-null    int64 
     1   Gender                  200 non-null    object
     2   Age                     200 non-null    int64 
     3   Annual Income (k$)      200 non-null    int64 
     4   Spending Score (1-100)  200 non-null    int64 
    dtypes: int64(4), object(1)
    memory usage: 7.9+ KB
    


```python
df.isnull().sum()
```




    CustomerID                0
    Gender                    0
    Age                       0
    Annual Income (k$)        0
    Spending Score (1-100)    0
    dtype: int64




```python
df.columns
```




    Index(['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
           'Spending Score (1-100)'],
          dtype='object')




```python
plt.figure(1, figsize=(15,6))
n = 0
for x in df.columns[2:]:
    n+=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    sns.distplot(df[x], bins=50)
    plt.title(x)
```


    
![png](img/output_9_0.png)
    



```python
plt.figure(1, figsize=(15,5))
sns.countplot(x='Gender', data=df)
plt.show()
```


    
![png](img/output_10_0.png)
    



```python
n = 1
plt.figure(1, figsize=(18, 4))
sns.set_theme(style="whitegrid")
for x in df.columns[2:]:
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.boxplot(x=x, y ='Gender', data=df)
    n+=1
plt.show()
```


    
![png](img/output_11_0.png)
    



```python
n=1
plt.figure(1, figsize=(15, 4))
for x in df.columns[2:]:
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.violinplot(x=x, y ='Gender', data=df)
    n+=1
plt.show()
```


    
![png](img/output_12_0.png)
    



```python
#sns.pairplot(hue='Gender', df[df.columns[2:]],)
data = df.drop(['CustomerID'], axis=1)
sns.pairplot(data, hue='Gender', diag_kind="hist")
plt.show()
```


    
![png](img/output_13_0.png)
    



```python
plt.figure(1, figsize=(15,6))
for gender in list(set(df.Gender)):
    plt.scatter(x='Spending Score (1-100)', y = 'Annual Income (k$)', data=df[df.Gender == gender], label=gender)
plt.title('Age vs Annual Income (k$)')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()
```


    
![png](img/output_14_0.png)
    


<h3 id='5'>Fit the cluster model</h3>


```python
X = df[['Age' , 'Spending Score (1-100)']].values
inertial = []
for n in range(1,11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.001, random_state=111, algorithm='elkan'))
    algorithm.fit(X)
    inertial.append(algorithm.inertia_)
```


```python
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
```


```python
plt.figure(1, figsize=(15, 4))
plt.plot(np.arange(1,11), inertia, 'o')
plt.plot(np.arange(1,11), inertia, '-')
plt.show()
```


    
![png](img/output_18_0.png)
    



```python
    algorithm = (KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit(X)
    labels1 = algorithm.labels_
    centroids1 = algorithm.cluster_centers_
```


```python
h = 0.02
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max, h))
Y = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
```


```python
plt.figure(1, figsize=(15,7))
plt.clf()
Y = Y.reshape(xx.shape)
plt.imshow(Y, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
plt.scatter(x='Age', y='Spending Score (1-100)', data=df, c=labels1)
plt.scatter(x=centroids1[:,0], y = centroids1[:,1], s=200, c = 'red', alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.show()
```


    
![png](img/output_21_0.png)
    



```python
X = df[['Annual Income (k$)' , 'Spending Score (1-100)']].values
inertia = []
for n in range(1,11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111,
                        algorithm='elkan'))    
    
    algorithm.fit(X)
    inertia.append(algorithm.inertia_)
```


```python
plt.figure(1, figsize=(15,7))
plt.plot(np.arange(1,11), inertia, 'o')
plt.plot(np.arange(1,11), inertia, '-')
plt.show()
```


    
![png](img/output_23_0.png)
    



```python
algorithm = (KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
```


```python
h = 0.2
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Y = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
```


```python
plt.figure(1, figsize=(15, 7))
plt.clf()
Y = Y.reshape(xx.shape)
plt.imshow(Y, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap = plt.cm.Pastel2,
           aspect='auto', origin='lower')
plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, c=labels2)

plt.show()
```


    
![png](img/output_26_0.png)
    



```python
X = df[df.columns[2:]].values
inertia = []
for n in range(1,11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        random_state=111, algorithm='elkan'))
    algorithm.fit(X)
    inertia.append(algorithm.inertia_)
```


```python
plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1,11), inertia, 'o')
plt.plot(np.arange(1,11), inertia, '-')
plt.show()
```


    
![png](img/output_28_0.png)
    



```python
algorithm = (KMeans(n_clusters = 6, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X)
labels3 = algorithm.labels_
centroid3 = algorithm.cluster_centers_
```


```python
df['labels'] = labels3
trace = go.Scatter3d(
    x = df['Age'],
    y = df['Spending Score (1-100)'],
    z = df['Annual Income (k$)'],
    mode='markers',
    marker=dict(
        color=df['labels'],
        size=10,
        line=dict(color=df['labels'], width=12),
        opacity=0.5
    )
)
data = [trace]
layout= go.Layout(
    title='Cluster',
    scene = dict(
        xaxis = dict(title = 'Age'),
        yaxis = dict(title = 'Spending Score'),
        zaxis = dict(title = 'Annual Income')
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
```


<div>                            <div id="5de46c58-079c-46d1-9199-563845085988" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("5de46c58-079c-46d1-9199-563845085988")) {                    Plotly.newPlot(                        "5de46c58-079c-46d1-9199-563845085988",                        [{"marker":{"color":[4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,2,5,2,0,4,5,2,0,0,0,2,0,0,2,2,2,2,2,0,2,2,0,2,2,2,0,2,2,0,0,2,2,2,2,2,0,2,0,0,2,2,0,2,2,0,2,2,0,0,2,2,0,2,0,0,0,2,0,2,0,0,2,2,0,2,0,2,2,2,2,2,0,0,0,0,0,2,2,2,2,0,0,0,3,0,3,1,3,1,3,1,3,0,3,1,3,1,3,1,3,1,3,0,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3],"line":{"color":[4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,2,5,2,0,4,5,2,0,0,0,2,0,0,2,2,2,2,2,0,2,2,0,2,2,2,0,2,2,0,0,2,2,2,2,2,0,2,0,0,2,2,0,2,2,0,2,2,0,0,2,2,0,2,0,0,0,2,0,2,0,0,2,2,0,2,0,2,2,2,2,2,0,0,0,0,0,2,2,2,2,0,0,0,3,0,3,1,3,1,3,1,3,0,3,1,3,1,3,1,3,1,3,0,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3],"width":12},"opacity":0.5,"size":10},"mode":"markers","type":"scatter3d","x":[19,21,20,23,31,22,35,23,64,30,67,35,58,24,37,22,35,20,52,35,35,25,46,31,54,29,45,35,40,23,60,21,53,18,49,21,42,30,36,20,65,24,48,31,49,24,50,27,29,31,49,33,31,59,50,47,51,69,27,53,70,19,67,54,63,18,43,68,19,32,70,47,60,60,59,26,45,40,23,49,57,38,67,46,21,48,55,22,34,50,68,18,48,40,32,24,47,27,48,20,23,49,67,26,49,21,66,54,68,66,65,19,38,19,18,19,63,49,51,50,27,38,40,39,23,31,43,40,59,38,47,39,25,31,20,29,44,32,19,35,57,32,28,32,25,28,48,32,34,34,43,39,44,38,47,27,37,30,34,30,56,29,19,31,50,36,42,33,36,32,40,28,36,36,52,30,58,27,59,35,37,32,46,29,41,30,54,28,41,36,34,32,33,38,47,35,45,32,32,30],"y":[39,81,6,77,40,76,6,94,3,72,14,99,15,77,13,79,35,66,29,98,35,73,5,73,14,82,32,61,31,87,4,73,4,92,14,81,17,73,26,75,35,92,36,61,28,65,55,47,42,42,52,60,54,60,45,41,50,46,51,46,56,55,52,59,51,59,50,48,59,47,55,42,49,56,47,54,53,48,52,42,51,55,41,44,57,46,58,55,60,46,55,41,49,40,42,52,47,50,42,49,41,48,59,55,56,42,50,46,43,48,52,54,42,46,48,50,43,59,43,57,56,40,58,91,29,77,35,95,11,75,9,75,34,71,5,88,7,73,10,72,5,93,40,87,12,97,36,74,22,90,17,88,20,76,16,89,1,78,1,73,35,83,5,93,26,75,20,95,27,63,13,75,10,92,13,86,15,69,14,90,32,86,15,88,39,97,24,68,17,85,23,69,8,91,16,79,28,74,18,83],"z":[15,15,16,16,17,17,18,18,19,19,19,19,20,20,20,20,21,21,23,23,24,24,25,25,28,28,28,28,29,29,30,30,33,33,33,33,34,34,37,37,38,38,39,39,39,39,40,40,40,40,42,42,43,43,43,43,44,44,46,46,46,46,47,47,48,48,48,48,48,48,49,49,50,50,54,54,54,54,54,54,54,54,54,54,54,54,57,57,58,58,59,59,60,60,60,60,60,60,61,61,62,62,62,62,62,62,63,63,63,63,63,63,64,64,65,65,65,65,67,67,67,67,69,69,70,70,71,71,71,71,71,71,72,72,73,73,73,73,74,74,75,75,76,76,77,77,77,77,78,78,78,78,78,78,78,78,78,78,78,78,79,79,81,81,85,85,86,86,87,87,87,87,87,87,88,88,88,88,93,93,97,97,98,98,99,99,101,101,103,103,103,103,113,113,120,120,126,126,137,137]}],                        {"scene":{"xaxis":{"title":{"text":"Age"}},"yaxis":{"title":{"text":"Spending Score"}},"zaxis":{"title":{"text":"Annual Income"}}},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Cluster"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('5de46c58-079c-46d1-9199-563845085988');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<h3 id='6'>Conclusion</h3>
<p>I think, we can create appropriate cluster using <b>Annual Income (k$)</b> and <b>Spending Score (1-100)</b></p>
<b>As we can see:</b>
<ul>
<li>The top left cluster represent low income but high spend</li>
<li>The top right cluster represent high income but high spend</li>
<li>The bottom left cluster represent low income but low spend</li>
<li>The bottom right cluster represent high income but low spend</li>
<li>The Middle cluster represent average income and spend</li>
</ul>

The combination of <b>Age</b>,<b>Annual Income (k$)</b> and <b>Spending Score (1-100)</b> is also perfect to cluster.


```python

```

#!/usr/bin/env python
# coding: utf-8

# <h1>Water Quality Analysis</h1>
# <p>Water Quality monitoring is definitely significant in our ecosystem. Poor quality water is more harmful to our daily life as well as aquatic  life, which could make a huge effect on the ecosystem.</p>
# <p>monitoring the quality of water can help researchers predict and learn from natural processes in the environment. Also, it could determine human impacts on an ecosystem.</p>
# <ul>
#     <li><a href="#2" style="text-decoration: none;">Why we are working about?</a></li>
#     <li><a href="#3" style="text-decoration: none;">What data we used?</a></li>
#     <li>
#         <ul>
#             <li><a href="#4" style="text-decoration: none;">Load Python libray and Data</a></li>
#             <li><a href="#5" style="text-decoration: none;">Preprocess and Anlysis</a></li>
#             <li><a href="#6" style="text-decoration: none;">Prediction</a></li>
#             <li><a href="#7" style="text-decoration: none;">Visualization</a></li>
#         </ul>
#     </li>
# </ul>
# 
# <h3 id='2'>Why we are predicting?</h3>
# <p>Since water is more and more essential thing for the animal kingdom, we need to take it more seriously to make it safe for animal life. Analysing water quality, humans could take steps to save the ecosystem. They can predict what should humans do. How to recover it and how to prepare for the next step. We know that global warming has become stressful for the world. Poor quality of water is one the reason.</p>
# 
# <h3 id='3'>What data is avaiable</h3>
# <p>To analys the water quality, we have used <a href="https://www.kaggle.com/adityakadiwal/water-potability">Kaggle Dataset</a>. Usign the dataset I will work on multiple collams. To understand all of the coulms you can go through the Kaggle <a href="https://www.kaggle.com/adityakadiwal/water-potability">Link</a>. For conveniant, I have used the same cntent for columns.</p>
# 
# <p><b>The water_potability.csv file contains water quality metrics for 3276 different water bodies.</b></p>
# 
# <h6>1. pH value:</h6>
# <p>PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.</p>
# <h6>2. Hardness:</h6>
# <p>Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.</p>
# <h6>3. Solids (Total dissolved solids - TDS):</h6>
# <p>Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.</p>
# <h6>4. Chloramines:</h6>
# <p>Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.</p>
# <h6>5. Sulfate:</h6>
# <p>Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.</p>
# <h6>6. Conductivity:</h6>
# <p>Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.
# </p>
# <h6>7. Organic_carbon:</h6>
# <p>Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.</p>
# <h6>8. Trihalomethanes:</h6>
# <p>THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.</p>
# <h6>9. Turbidity:</h6>
# <p>The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.</p>
# <h6>10. Potability:<h6>
# <p>Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.</p>
# 
# <h3 id='4'>Load Python libray and Data</h3>
# <p>After download the data. It time to load python libray.</p>

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df = pd.read_csv("data/water_potability.csv")
df.head()


# <h3 id='5'>Preprocessing</h3>

# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum() #To cehck out the null


# In[7]:


df = df.dropna()
df.isnull().sum()


# In[8]:


df.shape


# In[9]:


plt.figure(figsize=(10,5))
sns.countplot(df.Potability)
plt.title('Distribution of Unsafe and Safe Water') # 0 for safe and 1 unsafe
plt.show()


# In[10]:


df.Potability.value_counts().plot(kind ='pie')


# In[11]:


df.Potability.value_counts()


# <p>The sample is unbalance. the safe water is greater than unsafe. Since the number of sample of 0 is 1200 and 811 for 1, we need to upsample to balance it and to avoid the bias </p>

# In[12]:


from sklearn.utils import resample
zero = df[df['Potability'] == 0]
one = df[df['Potability'] == 1]
one_upsample = resample(one, replace = True, n_samples=1200)
df = pd.concat([zero, one_upsample])
from sklearn.utils import shuffle
df = shuffle(df)
df.shape


# In[13]:


df.Potability.value_counts().plot(kind='pie')


# In[14]:


plt.figure(figsize=(15,9))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot = True, cmap="Blues")
plt.show()


# <p><b>As we know the correlation range between -1 and 1. A perfect correlation by negative 1 or positive 1. 0 mean no correlation between the variable.</b></p>

# In[15]:


df.columns


# In[16]:


sns.pairplot(df, hue='Potability', height=2.5);
plt.show()


# In[17]:


df.corr()['Potability'].sort_values(ascending = False)


# <p><b>As I can see there less correlation between all factors</b></p>

# In[18]:


X = df.drop(['Potability'], axis = 1)
y = df['Potability']


# In[19]:


from sklearn.preprocessing import StandardScaler
stds = StandardScaler()
X[X.columns] = stds.fit_transform(X[X.columns])
df = pd.concat([X, y], axis=1)


# In[20]:


df.head()


# <h3 id='7'>Prediction</h3>

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print(X_train.shape)
print(X_test.shape)


# In[33]:


from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, confusion_matrix, plot_confusion_matrix
models = [('LR', LogisticRegression(max_iter=1000)),('Ridge', RidgeClassifier()),('SGDC', SGDClassifier()),('PassiveAggressive',PassiveAggressiveClassifier()),("SVC", SVC()), ('KNN', KNeighborsClassifier(n_neighbors=10))]
results = []
names = []
FinalResults = []
for name, model in models:
    model.fit(X_train,y_train)
    pred_y = model.predict(X_test)
    score = precision_score(y_test,pred_y)
    results.append(score)
    names.append(name)
    FinalResults.append((name, score))


# In[42]:


FinalResults


# <h4>We got higher accuracy for SVC which is 0.68 or 68%</h4>

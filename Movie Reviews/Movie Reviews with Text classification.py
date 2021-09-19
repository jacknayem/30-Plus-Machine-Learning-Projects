#!/usr/bin/env python
# coding: utf-8

# <h1>Sentiment Analysis</h1>
# <ul>
#     <li><a href="#1" style="text-decoration: none;">Why we to sentiment analyze?</a></li>
#     <li><a href="#2" style="text-decoration: none;">What data we used?</a></li>
#     <li>
#         <ul>
#             <li><a href="#3" style="text-decoration: none;">Load Python libray and data</a></li>
#             <li><a href="#6" style="text-decoration: none;">Text Preprocess</a></li>
#             <li><a href="#8" style="text-decoration: none;">Prediction</a></li>
#             <li><a href="#9" style="text-decoration: none;">Evaluation</a></li>
#         </ul>
#     </li>
# </ul>
# 
# <h3 id='1'>Why we to sentiment analyze?</h3>
# <p>Sentiment Analysis used to identify the toxic in specific public comment. In this analysis, I am trying to identify that is movie review toxic or natural.</p>
# 
# <h3 id='2'>Is data avaiable</h3>
# <p>Yes, the data is avaiable in NLTK corpus. I have used the movie review corpus.</p>
# 
# <h3 id='3'>Load Python libray and data</h3>

# In[1]:


import random
import nltk
from nltk.corpus import movie_reviews


# <h3 id='6'>Text Preprocess and pre-analysis</h3>

# In[2]:


movie_reviews.words()


# In[3]:


len(movie_reviews.words())


# In[4]:


movie_reviews.categories()


# In[5]:


nltk.FreqDist(movie_reviews.words())


# In[6]:


nltk.FreqDist(movie_reviews.words()).most_common(20)


# In[7]:


movie_reviews.fileids()


# In[8]:


movie_reviews.fileids('pos')


# In[9]:


movie_reviews.fileids('neg')


# In[10]:


movie_reviews.words('neg/cv018_21672.txt')


# In[11]:


all_words = nltk.FreqDist(movie_reviews.words())
len(all_words)


# In[12]:


feature_vector = list(all_words)[:4000]


# In[13]:


#Build the list of document
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories() 
             for fileid in movie_reviews.fileids(category)]

#Shuffle Document
random.shuffle(documents)


# In[14]:


documents[0]


# In[15]:


def find_feature(word_list):
    feature = {}
    for x in feature_vector:
        feature[x] = x in word_list
    return feature


# In[16]:


find_feature(documents[0][0])


# In[17]:


feature_sets = [(find_feature(word_list),category) for (word_list,category) in documents]


# <h3 id='8'>Prediction</h3>

# In[18]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn import model_selection


# In[19]:


train_set,test_set = model_selection.train_test_split(feature_sets,test_size = 0.25)
print(len(train_set))
print(len(test_set))


# In[27]:


model = SklearnClassifier(SVC(kernel = 'linear'))
model.train(train_set)


# <h3 id='9'>Evaluation</h3>

# In[26]:


accuracy = nltk.classify.accuracy(model, test_set)
print('SVC Accuracy : {}'.format(accuracy))


# In[ ]:





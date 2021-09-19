#!/usr/bin/env python
# coding: utf-8

# <h1>Toxic Comments Analysis</h1>
# <ul>
#     <li><a href="#1" style="text-decoration: none;">Why we are working about?</a></li>
#     <li><a href="#2" style="text-decoration: none;">What data we used?</a></li>
#     <li>
#         <ul>
#             <li><a href="#3" style="text-decoration: none;">Load Python libray</a></li>
#             <li><a href="#4" style="text-decoration: none;">Data</a></li>
#             <li><a href="#5" style="text-decoration: none;">Preprocess and Visualization</a></li>
#             <li><a href="#6" style="text-decoration: none;">Text Preprocess</a></li>
#             <li><a href="#7" style="text-decoration: none;">Feature Extraction</a></li>
#             <li><a href="#8" style="text-decoration: none;">Prediction</a></li>
#             <li><a href="#9" style="text-decoration: none;">Evaluation</a></li>
#         </ul>
#     </li>
# </ul>
# 
# <h3 id='1'>Why we are predicting?</h3>
# <p>Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.</p>
# 
# <h3 id='2'>What data is avaiable</h3>
# <p>To analys the toxic comment, we have used <a href="https://www.kaggle.com/c/nlp-getting-started/data">Kaggle Dataset</a>.</p>
# 
# <h3 id='3'>Load Python libray</h3>

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from lightgbm import LGBMClassifier
import pandas as pd
import re


# <h3 id='4'>Load Data</h3>

# In[17]:


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


# In[18]:


train_df.head()


# In[19]:


test_df.head()


# In[20]:


print(train_df.shape)
print(test_df.shape)


# <h3 id='5'>Preprocess and pre-analysis</h3>

# In[23]:


train_df.info()


# In[24]:


train_df.describe()


# In[26]:


train_df = train_df.drop(['id','keyword','location'], axis=1)
test_df = test_df.text


# <h3 id='6'>Text Preprocess</h3>

# In[29]:


# Create small function to clean text
def text_clean(text):

    for element in ["http\S+", "RT ", "[^a-zA-Z\'\.\,\d\s]", "[0-9]","\t", "\n", "\s+", "<.*?>"]:
        text = re.sub("r"+element, " ", text)

    return text

# Clean data sets
train_df.text = train_df.text.apply(text_clean)
test_df.text = val_df.text.apply(text_clean)
print('Done')


# In[30]:


# CORRECT SPELLING
# Instantiate spell checker
spell = SpellChecker()

# Correct spelling
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
# Spellcheck data sets
train_df.text = train_df.text.apply(correct_spellings)
test_df.text = val_df.text.apply(correct_spellings)
print('Done')


# In[32]:


train_df.head()


# In[65]:


X = train_df.text
y = train_df.target


# <h3 id='7'>Feature Extraction</h3>

# In[66]:


vectorizer = TfidfVectorizer(max_features=1500, min_df = 5, max_df=0.7, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(X).toarray()


# <h3 id='8'>Train and Prediction</h3>

# In[67]:


# Split data into train and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

clf = LGBMClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)


# <h3 id='9'>Evaluation</h3>

# In[72]:


accuracy = accuracy_score(y_pred, y_test)
print('LightGBM Accuracy Score: ',accuracy)


# In[73]:


con_met = confusion_matrix(y_pred, y_test)
con_met


# In[69]:


y_pred_train=clf.predict(X_train)
accuracy_train = accuracy_score(y_pred_train, y_train)
print('LightGBM Accuracy Score: ',accuracy_train)


# In[70]:


print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))


# In[77]:


from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test))


# In[ ]:





<h1>Toxic Comments Analysis</h1>
<ul>
    <li><a href="#1" style="text-decoration: none;">Why we are working about?</a></li>
    <li><a href="#2" style="text-decoration: none;">What data we used?</a></li>
    <li>
        <ul>
            <li><a href="#3" style="text-decoration: none;">Load Python libray</a></li>
            <li><a href="#4" style="text-decoration: none;">Data</a></li>
            <li><a href="#5" style="text-decoration: none;">Preprocess and Visualization</a></li>
            <li><a href="#6" style="text-decoration: none;">Text Preprocess</a></li>
            <li><a href="#7" style="text-decoration: none;">Feature Extraction</a></li>
            <li><a href="#8" style="text-decoration: none;">Prediction</a></li>
            <li><a href="#9" style="text-decoration: none;">Evaluation</a></li>
        </ul>
    </li>
</ul>

<h3 id='1'>Why we are predicting?</h3>
<p>Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.</p>

<h3 id='2'>What data is avaiable</h3>
<p>To analys the toxic comment, we have used <a href="https://www.kaggle.com/c/nlp-getting-started/data">Kaggle Dataset</a>.</p>

<h3 id='3'>Load Python libray</h3>


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from lightgbm import LGBMClassifier
import pandas as pd
import re
```

<h3 id='4'>Load Data</h3>


```python
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
```


```python
train_df.head()
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
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
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
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(train_df.shape)
print(test_df.shape)
```

    (7613, 5)
    (3263, 4)
    

<h3 id='5'>Preprocess and pre-analysis</h3>


```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7613 entries, 0 to 7612
    Data columns (total 5 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   id        7613 non-null   int64 
     1   keyword   7552 non-null   object
     2   location  5080 non-null   object
     3   text      7613 non-null   object
     4   target    7613 non-null   int64 
    dtypes: int64(2), object(3)
    memory usage: 297.5+ KB
    


```python
train_df.describe()
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
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7613.000000</td>
      <td>7613.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5441.934848</td>
      <td>0.42966</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3137.116090</td>
      <td>0.49506</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2734.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5408.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8146.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10873.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df = train_df.drop(['id','keyword','location'], axis=1)
test_df = test_df.text
```

<h3 id='6'>Text Preprocess</h3>


```python
# Create small function to clean text
def text_clean(text):

    for element in ["http\S+", "RT ", "[^a-zA-Z\'\.\,\d\s]", "[0-9]","\t", "\n", "\s+", "<.*?>"]:
        text = re.sub("r"+element, " ", text)

    return text

# Clean data sets
train_df.text = train_df.text.apply(text_clean)
test_df.text = val_df.text.apply(text_clean)
print('Done')
```

    Done
    


```python
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
```

    Done
    


```python
train_df.head()
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
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ou Deeds are the Reason of this earthquake May...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Forest fire nea La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All residents asked to shelter in place are be...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13,000 people receive wildfires evacuation ord...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = train_df.text
y = train_df.target
```

<h3 id='7'>Feature Extraction</h3>


```python
vectorizer = TfidfVectorizer(max_features=1500, min_df = 5, max_df=0.7, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(X).toarray()
```

<h3 id='8'>Train and Prediction</h3>


```python
# Split data into train and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

clf = LGBMClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
```

<h3 id='9'>Evaluation</h3>


```python
accuracy = accuracy_score(y_pred, y_test)
print('LightGBM Accuracy Score: ',accuracy)
```

    LightGBM Accuracy Score:  0.7800393959290873
    


```python
con_met = confusion_matrix(y_pred, y_test)
con_met
```




    array([[752, 213],
           [122, 436]], dtype=int64)




```python
y_pred_train=clf.predict(X_train)
accuracy_train = accuracy_score(y_pred_train, y_train)
print('LightGBM Accuracy Score: ',accuracy_train)
```

    LightGBM Accuracy Score:  0.8596059113300493
    


```python
print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))
```

    Training set score: 0.8596
    Test set score: 0.7800
    


```python
from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.78      0.82       965
               1       0.67      0.78      0.72       558
    
        accuracy                           0.78      1523
       macro avg       0.77      0.78      0.77      1523
    weighted avg       0.79      0.78      0.78      1523
    
    


```python

```

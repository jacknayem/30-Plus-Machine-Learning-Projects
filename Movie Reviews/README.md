<h1>Sentiment Analysis</h1>
<ul>
    <li><a href="#1" style="text-decoration: none;">Why we to sentiment analyze?</a></li>
    <li><a href="#2" style="text-decoration: none;">What data we used?</a></li>
    <li>
        <ul>
            <li><a href="#3" style="text-decoration: none;">Load Python libray and data</a></li>
            <li><a href="#6" style="text-decoration: none;">Text Preprocess</a></li>
            <li><a href="#8" style="text-decoration: none;">Prediction</a></li>
            <li><a href="#9" style="text-decoration: none;">Evaluation</a></li>
        </ul>
    </li>
</ul>

<h3 id='1'>Why we to sentiment analyze?</h3>
<p>Sentiment Analysis used to identify the toxic in specific public comment. In this analysis, I am trying to identify that is movie review toxic or natural.</p>

<h3 id='2'>Is data avaiable</h3>
<p>Yes, the data is avaiable in NLTK corpus. I have used the movie review corpus.</p>

<h3 id='3'>Load Python libray and data</h3>


```python
import random
import nltk
from nltk.corpus import movie_reviews
```

<h3 id='6'>Text Preprocess and pre-analysis</h3>


```python
movie_reviews.words()
```




    ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', ...]




```python
len(movie_reviews.words())
```




    1583820




```python
movie_reviews.categories()
```




    ['neg', 'pos']




```python
nltk.FreqDist(movie_reviews.words())
```




    FreqDist({',': 77717, 'the': 76529, '.': 65876, 'a': 38106, 'and': 35576, 'of': 34123, 'to': 31937, "'": 30585, 'is': 25195, 'in': 21822, ...})




```python
nltk.FreqDist(movie_reviews.words()).most_common(20)
```




    [(',', 77717),
     ('the', 76529),
     ('.', 65876),
     ('a', 38106),
     ('and', 35576),
     ('of', 34123),
     ('to', 31937),
     ("'", 30585),
     ('is', 25195),
     ('in', 21822),
     ('s', 18513),
     ('"', 17612),
     ('it', 16107),
     ('that', 15924),
     ('-', 15595),
     (')', 11781),
     ('(', 11664),
     ('as', 11378),
     ('with', 10792),
     ('for', 9961)]




```python
movie_reviews.fileids()
```




    ['neg/cv000_29416.txt',
     'neg/cv001_19502.txt',
     'neg/cv002_17424.txt',
     'neg/cv003_12683.txt',
     'neg/cv004_12641.txt',
     .........
     'neg/cv994_13229.txt',
     'neg/cv995_23113.txt',
     'neg/cv996_12447.txt',
     'neg/cv997_5152.txt',
     'neg/cv998_15691.txt',
     'neg/cv999_14636.txt']




```python
movie_reviews.words('neg/cv018_21672.txt')
```




    ['the', 'law', 'of', 'crowd', 'pleasing', 'romantic', ...]




```python
all_words = nltk.FreqDist(movie_reviews.words())
len(all_words)
```




    39768




```python
feature_vector = list(all_words)[:4000]
```


```python
#Build the list of document
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories() 
             for fileid in movie_reviews.fileids(category)]

#Shuffle Document
random.shuffle(documents)
```


```python
documents[0]
```




    (['guilt',
      '.',
      'guilt',
      'is',
      'something',
      'i',
      'felt',
      'while',
      'watching',
      'basic',
      'instinct',
      'for',
      'the',
      'ninth',
      'time',
      ';',
      'the',
      'penultimate',
      ......
      'starship',
      'troopers',
      'if',
      'they',
      'could',
      'see',
      'it',
      '.'],
     'neg')




```python
def find_feature(word_list):
    feature = {}
    for x in feature_vector:
        feature[x] = x in word_list
    return feature
```


```python
find_feature(documents[0][0])
```




    {',': True,
     'the': True,
     '.': True,
     'a': True,
     .........
     'sci': False,
     'wait': False,
     'sit': False,
     'female': True,
     ...}




```python
feature_sets = [(find_feature(word_list),category) for (word_list,category) in documents]
```

<h3 id='8'>Prediction</h3>


```python
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn import model_selection
```


```python
train_set,test_set = model_selection.train_test_split(feature_sets,test_size = 0.25)
print(len(train_set))
print(len(test_set))
```

    1500
    500
    


```python
model = SklearnClassifier(SVC(kernel = 'linear'))
model.train(train_set)
```




    <SklearnClassifier(SVC(kernel='linear'))>



<h3 id='9'>Evaluation</h3>


```python
accuracy = nltk.classify.accuracy(model, test_set)
print('SVC Accuracy : {}'.format(accuracy))
```

    SVC Accuracy : 0.838
    


```python

```

## Rotten Tomatoes Movies Rating Prediction

### Table of Contents
1. First Approach: Predicting Movie Status Based on Numerical and Categorical Features
   - Data Preprocessing
   - Random Forest Classifier
   - Random Forest Classifier with Feature Selection
   - Weighted Random Forest Classifier with Feature Selection
2. Second Approach: Predicting Movie Status Based on Review Sentiment
   - Default Random Forest
   - Weighted Random Forest
   - Movie Status Prediction

### First Approach: Predicting Movie Status Based on Numerical and Categorical Features

#### Data Preprocessing

We start by loading the dataset. To do that, we import the necessary libraries and then use pandas to read the df_movies csv file.

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
```python

```


```python
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
```


```python
df_movies = pd.read_csv('./datasets/rotten_tomatoes_movies.csv')
```


```python
df_reviews = pd.read_csv('./datasets/rotten_tomatoes_critic_reviews_50k.csv')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 1
    ----> 1 df_reviews = pd.read_csv('./datasets/rotten_tomatoes_critic_reviews_50k.csv')
    

    NameError: name 'pd' is not defined



```python
df_movies.head(3)
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
      <th>rotten_tomatoes_link</th>
      <th>movie_title</th>
      <th>movie_info</th>
      <th>critics_consensus</th>
      <th>content_rating</th>
      <th>genres</th>
      <th>directors</th>
      <th>authors</th>
      <th>actors</th>
      <th>original_release_date</th>
      <th>streaming_release_date</th>
      <th>runtime</th>
      <th>production_company</th>
      <th>tomatometer_status</th>
      <th>tomatometer_rating</th>
      <th>tomatometer_count</th>
      <th>audience_status</th>
      <th>audience_rating</th>
      <th>audience_count</th>
      <th>tomatometer_top_critics_count</th>
      <th>tomatometer_fresh_critics_count</th>
      <th>tomatometer_rotten_critics_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>m/0814255</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>Always trouble-prone, the life of teenager Per...</td>
      <td>Though it may seem like just another Harry Pot...</td>
      <td>PG</td>
      <td>Action &amp; Adventure, Comedy, Drama, Science Fic...</td>
      <td>Chris Columbus</td>
      <td>Craig Titley, Chris Columbus, Rick Riordan</td>
      <td>Logan Lerman, Brandon T. Jackson, Alexandra Da...</td>
      <td>2010-02-12</td>
      <td>2015-11-25</td>
      <td>119.0</td>
      <td>20th Century Fox</td>
      <td>Rotten</td>
      <td>49.0</td>
      <td>149.0</td>
      <td>Spilled</td>
      <td>53.0</td>
      <td>254421.0</td>
      <td>43</td>
      <td>73</td>
      <td>76</td>
    </tr>
    <tr>
      <th>1</th>
      <td>m/0878835</td>
      <td>Please Give</td>
      <td>Kate (Catherine Keener) and her husband Alex (...</td>
      <td>Nicole Holofcener's newest might seem slight i...</td>
      <td>R</td>
      <td>Comedy</td>
      <td>Nicole Holofcener</td>
      <td>Nicole Holofcener</td>
      <td>Catherine Keener, Amanda Peet, Oliver Platt, R...</td>
      <td>2010-04-30</td>
      <td>2012-09-04</td>
      <td>90.0</td>
      <td>Sony Pictures Classics</td>
      <td>Certified-Fresh</td>
      <td>87.0</td>
      <td>142.0</td>
      <td>Upright</td>
      <td>64.0</td>
      <td>11574.0</td>
      <td>44</td>
      <td>123</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>m/10</td>
      <td>10</td>
      <td>A successful, middle-aged Hollywood songwriter...</td>
      <td>Blake Edwards' bawdy comedy may not score a pe...</td>
      <td>R</td>
      <td>Comedy, Romance</td>
      <td>Blake Edwards</td>
      <td>Blake Edwards</td>
      <td>Dudley Moore, Bo Derek, Julie Andrews, Robert ...</td>
      <td>1979-10-05</td>
      <td>2014-07-24</td>
      <td>122.0</td>
      <td>Waner Bros.</td>
      <td>Fresh</td>
      <td>67.0</td>
      <td>24.0</td>
      <td>Spilled</td>
      <td>53.0</td>
      <td>14684.0</td>
      <td>2</td>
      <td>16</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_movies.describe().columns
```




    Index(['runtime', 'tomatometer_rating', 'tomatometer_count', 'audience_rating',
           'audience_count', 'tomatometer_top_critics_count',
           'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count'],
          dtype='object')




```python
numeric_columns = df_movies.describe().columns.tolist()
categorical_columns = ['content_rating'] 
# , 'genres'
ordinal_columns = ['audience_status']
target = 'tomatometer_status'
features = numeric_columns + categorical_columns + ordinal_columns
```


```python
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])

preprocessor = ColumnTransformer(
            transformers =[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns),
                ('ord', ordinal_transformer, ordinal_columns)   
            ])

rf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])
```


```python
df_no_na = df_movies.dropna(subset=features + [target])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_no_na[features], df_no_na[target], test_size=0.2, random_state=42)
```


```python

rf.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = (y_pred == y_test).mean()
# accuracy_score = tree.score(X_test, y_test)
accuracy
```




    0.9905992949471211




```python
print(classification_report(y_test, y_pred))
```

                     precision    recall  f1-score   support
    
    Certified-Fresh       0.98      0.97      0.97       630
              Fresh       0.99      0.99      0.99      1286
             Rotten       1.00      1.00      1.00      1488
    
           accuracy                           0.99      3404
          macro avg       0.99      0.99      0.99      3404
       weighted avg       0.99      0.99      0.99      3404
    
    


```python
# Get the feature importance
# feature_importance = rf['classifier'].feature_importances_


preprocessor = rf.named_steps['preprocessor']

# Get feature names after transformation
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'drop' or transformer == 'passthrough':
            continue
        if hasattr(transformer, 'named_steps'):
            for step in transformer.named_steps.values():
                if hasattr(step, 'get_feature_names_out'):
                    feature_names.extend(step.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    return feature_names


# Print feature importance
for i, feature in enumerate(X_train.columns):
    print(f'{feature} = {feature_importance[i]}')

# Get all feature names
feature_names = get_feature_names(preprocessor)

# Get feature importances from the classifier
feature_importance = rf.named_steps['classifier'].feature_importances_

# Visualize feature from the most important to the least important
indices = np.argsort(feature_importance)

plt.figure(figsize=(12,9))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

    runtime = 0.0058771986409833265
    tomatometer_rating = 0.5030182802392604
    tomatometer_count = 0.10924088979195079
    audience_rating = 0.0508781482208602
    audience_count = 0.018134119844937294
    tomatometer_top_critics_count = 0.03642304051758556
    tomatometer_fresh_critics_count = 0.12257226216398548
    tomatometer_rotten_critics_count = 0.11374929064388548
    content_rating = 0.00035198243716457345
    audience_status = 3.236616551134634e-05
    


    
![png](Rating_Prediction_Models_files/Rating_Prediction_Models_11_1.png)
    



```python

```


```python

```


```python

```


# There are many other languages you can use, but the style has to be loaded first

ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```

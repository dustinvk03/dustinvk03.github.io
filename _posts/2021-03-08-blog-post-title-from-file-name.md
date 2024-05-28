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

# Load dataset
df_movies = pd.read_csv('./datasets/rotten_tomatoes_movies.csv')
```
Then we can define the features
```python
# Define feature categories
numeric_columns = df_movies.describe().columns.tolist()
categorical_columns = ['content_rating']
ordinal_columns = ['audience_status']
target = 'tomatometer_status'
```
Preprocessing pipelines pipeline with corresponding datatypes:
```python
# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns),
        ('ord', ordinal_transformer, ordinal_columns)
    ]
)
```
# Clean dataset
df_no_na = df_movies.dropna(subset=numeric_columns + categorical_columns + ordinal_columns + [target])

  
We start by loading and preprocessing the dataset.

---

### This is a header

#### Some T-SQL Code

```tsql
SELECT This, [Is], A, Code, Block -- Using SSMS style syntax highlighting
    , REVERSE('abc')
FROM dbo.SomeTable s
    CROSS JOIN dbo.OtherTable o;
```

#### Some PowerShell Code

```powershell
Write-Host "This is a powershell Code block";

# There are many other languages you can use, but the style has to be loaded first

ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```

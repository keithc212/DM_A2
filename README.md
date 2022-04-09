import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier

# replace any NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(0)

# select features and labels
x = df.shape[1]-1
features = df.iloc[:,0:x]
labels = df["target"]

# split data for cross-validation before final evaluation
train_features, test_features, train_labels, test_labels = model_selection.train_test_split(features, labels, test_size=0.3, random_state=0)

# using a pipeline, scale  the data and classify it
cls = make_pipeline(StandardScaler(), SGDClassifier())

# fit classifier to data
cls.fit(train_features, train_labels)

# create predictions and score accuracy
print(cls.score(test_features, test_labels))

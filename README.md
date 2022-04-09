import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(0)

x = df.shape[1]-1
features = df.iloc[:,0:x]
labels = df["target"]

train_features, test_features, train_labels, test_labels = model_selection.train_test_split(features, labels, test_size=0.3, random_state=0)

cls = make_pipeline(StandardScaler(), SGDClassifier())

cls.fit(train_features, train_labels)

print(cls.score(test_features, test_labels))


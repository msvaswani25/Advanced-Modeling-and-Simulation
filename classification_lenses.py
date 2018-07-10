# -*- coding: utf-8 -*-
"""
@author: MS Vaswani
"""
#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Loading dataset
input_file = "lenses.csv"
data = pd.read_csv(input_file, header = 0, delimiter="\t")
#
# Organizing data
classes = data['class']
attributes = data.loc[:,:'tear_production_rate']

#spliting data into train and test
train, test, train_labels, test_labels = train_test_split(attributes,
                                                          classes,
                                                          test_size=0.33,
                                                          random_state=42)

# Initializing classifier model
gnb = GaussianNB()

# Training classifier model
model = gnb.fit(train, train_labels)

# predicting
predection = gnb.predict(test)
print(predection)

# Evaluating accuracy
accuracy=accuracy_score(test_labels, predection)
print(accuracy*100,"%")
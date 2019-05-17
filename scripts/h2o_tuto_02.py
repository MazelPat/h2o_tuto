# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:33:47 2018

@author: pmazel
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
# import numpy as np

h2o.init(nthreads=2, max_mem_size=8)

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("../data/train_reduit.csv")
test = h2o.import_file("../data/test_survived.csv")

# Identify predictors and response
x = train.columns
y = "Survived"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 60)
aml.train(x = x, y = y,
          training_frame = train)

# View the AutoML Leaderboard
lb = aml.leaderboard

# The leader model is stored here
aml.leader

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)

h2o.export_file(preds, path = "../data/pred_Python.csv")

h2o.cluster().shutdown()

testcsv = pd.read_csv("../data/test.csv")
predcsv = pd.read_csv("../data/pred_Python.csv")

testpred = pd.concat([testcsv, predcsv], axis=1, ignore_index=False) 
#columns=["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked","predict","p0","p1"]

testpred.to_csv("../data/predictions_python.csv")
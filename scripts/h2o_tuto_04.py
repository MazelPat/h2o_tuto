# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:09:42 2018

@author: pmazel
"""

import h2o
from h2o.estimators import H2OXGBoostEstimator

h2o.init(nthreads=2, max_mem_size=4)
# In the next step, we import and prepare data via the H2O API:

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

# Nombre aléatoire
r = train.runif()   

# all values of r under .8 are assigned to 'train_split' (80 percent)
train_split = train[r  < 0.8]
# all values of r equal or above .8 are assigned 'valid_split' (20 percent)
valid_split = train[r >= 0.8]


param = {
      "ntrees" : 100
    , "max_depth" : 10
    , "learn_rate" : 0.02
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100
}

model = H2OXGBoostEstimator(**param)

model.train(
        x = x,
        y = y,
        training_frame = train_split,
        validation_frame = valid_split)

# Perform classification on the test set (predict the handwritten digit, given the test set, which is unlabeled)
pred = model.predict(test)

# Take a look at the predictions
print(pred.head())

# h2o.cluster().shutdown()

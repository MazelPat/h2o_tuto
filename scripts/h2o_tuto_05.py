# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:09:42 2018

@author: pmazel
"""

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init(nthreads = 4, max_mem_size = 8)
# In the next step, we import and prepare data via the H2O API:

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("../data/fashion-mnist_train.csv")
test = h2o.import_file("../data/fashion-mnist_test.csv")

# Identify predictors and response
x = train.columns
y = "label"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Nombre aléatoire
r = train.runif()   
print(r)

# all values of r under .8 are assigned to 'train_split' (80 percent)
train_split = train[r  < 0.8]
# all values of r equal or above .8 are assigned 'valid_split' (20 percent)
valid_split = train[r >= 0.8]

# set the parameters of your Deep Learning model
model = H2ODeepLearningEstimator(
        distribution="multinomial",
        activation="RectifierWithDropout",
        hidden=[128,128,256],
        input_dropout_ratio=0.2, 
        sparse=True, 
        l1=1e-5, 
        epochs=200)

# train your model on your train_split data set, and then validate it on the valid_split set
# x is the list of column names excluding the response column (all of your features)
# y is the name of the response column ('label')
model.train(
        x = x, 
        y = y,
        training_frame=train_split, 
        validation_frame=valid_split)

# Perform classification on the test set (predict the handwritten digit, given the test set, which is unlabeled)
pred = model.predict(test)

# Take a look at the predictions
print(pred.head())

# h2o.cluster().shutdown()

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:09:42 2018

@author: pmazel
"""

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Import H2O Grid Search:
from h2o.grid.grid_search import H2OGridSearch

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

# DL hyperparameters
activation_opt = ["Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout"]
l1_opt = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
l2_opt = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
dl_params = {'activation': activation_opt, 'l1': l1_opt, 'l2': l2_opt}

# Search criteria
search_criteria = {'strategy': 'RandomDiscrete', 'max_runtime_secs': 1200, 'seed':1}

dl_grid = H2OGridSearch(model=H2ODeepLearningEstimator,
                        grid_id='dl_grid1',
                        hyper_params=dl_params,
                        search_criteria=search_criteria)

dl_grid.train(x=x, y=y,
              training_frame=train_split, 
              validation_frame=valid_split, 
              hidden=[100,50,10],
              search_criteria=search_criteria)

dl_gridperf = dl_grid.get_grid(sort_by='rmse', decreasing=True)

# Grab the model_id for the top GBM model, chosen by validation RMSE
best_dl_model = dl_gridperf.models[0]

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance

dl_perf = best_dl_model.model_performance(test)
print('performance : ', dl_perf.rmse())


# h2o.cluster().shutdown()

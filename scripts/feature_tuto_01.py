# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 13:03:03 2018

@author: pmazel
"""

import featuretools as ft
import numpy as np
import pandas as pd

train = pd.read_csv("../data/Train_UWu5bXk.csv")
test = pd.read_csv("../data/Test_u94Q5KV.csv")

# saving identifiers
test_Item_Identifier = test['Item_Identifier']
test_Outlet_Identifier = test['Outlet_Identifier']
sales = train['Item_Outlet_Sales']

train.drop(['Item_Outlet_Sales'], axis=1, inplace=True)

combi = train.append(test, ignore_index=True)

# print(combi.isnull().sum())

# imputing missing data
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace = True)
combi['Outlet_Size'].fillna("missing", inplace = True)

print(combi['Item_Fat_Content'].value_counts())

# dictionary to replace the categories
fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}

combi['Item_Fat_Content'] = combi['Item_Fat_Content'].replace(fat_content_dict, regex=True)

print(combi['Item_Fat_Content'].value_counts())

combi['id'] = combi['Item_Identifier'] + combi['Outlet_Identifier']
combi.drop(['Item_Identifier'], axis=1, inplace=True)

# creating and entity set 'es'
es = ft.EntitySet(id = 'sales')

# adding a dataframe 
es.entity_from_dataframe(entity_id = 'bigmart', dataframe = combi, index = 'id')

es.normalize_entity(base_entity_id='bigmart', new_entity_id='outlet', index = 'Outlet_Identifier', 
additional_variables = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

print(es)

feature_matrix, feature_names = ft.dfs(entityset=es, 
target_entity = 'bigmart', 
max_depth = 2, 
verbose = 1, 
n_jobs = 1)


print(feature_matrix.columns)
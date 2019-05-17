# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:20:54 2018

@author: pmazel
"""

import tensorflow as tf

sess = tf.Session()

print(tf.__version__)

a = tf.ones((2, 2))
b = tf.matmul(a, a)
b.eval(session=sess)


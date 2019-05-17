# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:28:49 2018

@author: pmazel
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import randint


np.random.seed(123)  # for reproducibility

print('Tensorflow version : ', tf.__version__)

mnist = tf.keras.datasets.mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# plt.imshow(x_train[2], cmap = plt.cm.binary)
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 1, batch_size = 128,
                    verbose = 1, validation_split = 0.05)

print(model.summary())

history = model.fit(x_train, y_train, epochs = 20, batch_size = 128,
                    verbose = 1, validation_split = 0.15)
 
# list all data in history
# print(history.history.keys())

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

predictions = model.predict([x_test])

plt.figure(1, figsize=(15, 15))

# summarize history for accuracy
plt.subplot(221)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
# plt.show()

# summarize history for loss
plt.subplot(222)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# print(predictions)

title = []
position = []
plt.figure(1, figsize=(15, 15))

i = 0

while i < 8:
    j = randint(0,9999)
    if  (y_test[j] == np.argmax(predictions[j])):
        etat = 'OK'
    else:
        etat = 'KO'
        title.append('Prédic. : ' + str(np.argmax(predictions[j])) 
        + ' - Réel : ' + str(y_test[j]) + ' => ' + etat)
        position.append(j)
        plt.subplot('44'+str(i+1))
        plt.title(title[i])
        plt.imshow(x_test[position[i]], cmap = plt.cm.binary)
        i += 1

plt.show()


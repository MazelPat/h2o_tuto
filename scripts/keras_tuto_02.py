# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:28:27 2018

@author: pmazel
"""

# 3. Import libraries and modules
import numpy as np
import theano as th
import keras
np.random.seed(123)  # for reproducibility

print(th.__version__)
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import matplotlib.pyplot as plt

print(keras.__version__)
 
# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# 5. Preprocess input data
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
 
# 6. Preprocess class labels
X_train = np_utils.normalize(X_train, axis=1)
X_test = np_utils.normalize(X_test, axis=1)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
# 7. Define model architecture
model = Sequential()
 
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
history = model.fit(X_train, Y_train, 
          batch_size=128, epochs=100, verbose=1)
 
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Score : ', score)

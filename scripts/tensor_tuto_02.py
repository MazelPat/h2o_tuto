import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Dense

print(tf.__version__)

# number iterations
ITER = 4000

# X = input of our 3 input XOR gate
# set up the inputs of the neural network (right from the table)
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0],
    [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]), dtype=float)
# y = our output of our neural network
y = np.array(([1], [0], [0], [0], [0],
    [0], [0], [1]), dtype=float)


#sample dataset
X = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
print("The shape of X is: ", X.shape)

model = tf.keras.Sequential()

model.add(Dense(32, input_dim=1, activation=tf.nn.sigmoid, use_bias=True))
model.add(Dense(32, activation='sigmoid', use_bias=True))
model.add(Dense(16, activation='relu', use_bias=True))
#model.add(Dense(16, activation='relu', use_bias=True))
model.add(Dense(4, activation='relu', use_bias=True))

model.add(Dense(1, activation='sigmoid', use_bias=True))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

print (model.get_weights())

history = model.fit(X, y, epochs=ITER, validation_data=(X, y))

model.summary()

print (model.get_weights())

# printing out to file
loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)

np.savetxt("loss_history.txt", numpy_loss_history, delimiter="\n")
binary_accuracy_history = history.history["binary_accuracy"]
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary_accuracy.txt", numpy_binary_accuracy, delimiter="\n")

print(np.mean(history.history["binary_accuracy"]))

result = model.predict(X).round()

print(result)

x1 = np.linspace(1, ITER, ITER)

fig, ax = plt.subplots(nrows=2)

ax[0].plot(x1, numpy_loss_history, color="blue", label="loss(iter)")
ax[0].set_xlabel("iter")
ax[0].set_ylabel("loss")
ax[0].legend()

ax[1].plot(x1, numpy_binary_accuracy, color="red", label="accuracy(iter)")
ax[1].set_xlabel("iter")
ax[1].set_ylabel("accuracy")
ax[1].legend()

plt.show()


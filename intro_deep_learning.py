import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

model = tf.keras.models.Sequential()
# Add a layer that flattens the input image
#model.add(tf.keras.layers.Flatten())
# Add a layer that is densely-connected: Which means that the nodes from the previous layers are all connected.
# The activation function relu means rectified linear
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=x_train.shape[1:]))
# Add second layer that is densely connected
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Add output layer that has one node
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
val_loss, val_accuracy = model.evaluate(x_test, y_test)
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import cv2
import matplotlib.pyplot as plt


celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} градусов Цельсия = {} градусов Фаренгейта".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
# ll0 = keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(lr=0.05))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Training complete")

plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show()

print(model.predict([100.0]))
print("Это значения весов l0: {}".format(l0.get_weights()))
print("Это значения весов l1: {}".format(l1.get_weights()))
print("Это значения весов l2: {}".format(l2.get_weights()))

cap = cv2.VideoCapture(0)
assert cap.isOpened()
_, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
try:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
except cv2.error:
    pass
cv2.imshow("Image", frame)
cv2.waitKey(0)

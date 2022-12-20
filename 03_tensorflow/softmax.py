import keras.losses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_blobs
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# SOFTMAX
# centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
# X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
#
# model = Sequential(
#     [
#         Dense(25, activation= 'relu'),
#         Dense(15, activation= 'relu'),
#         Dense(4, activation= 'linear') # note
#     ]
# )
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=adam_v2.Adam(learning_rate=0.001)
# )
#
# model.fit(
#     X_train, y_train,
#     epochs=10
# )
#
# p_preferred = model.predict(X_train)
# # result is not probability
# print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))
#
# for i in range(5):
#     print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")


# MULTICLASS
# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation='relu',   name="L1"),
        Dense(4, activation='linear', name="L2")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=adam_v2.Adam(learning_rate=0.01),
)

model.fit(
    X_train, y_train,
    epochs=200)

"""
Warm-up question

You are given two arrays (features and targets) as a dataset. If you take a closer look at values
inside, you'll notice that those replicate the function -> f(x) = X^2

Create a neural network that can will learn this function.

To test your model you can call the predict method on it, for example: model.predict([7.0])
(The expected result here is 49.0)

"""

import numpy as np

import tensorflow as tf


def solution_model():
	features = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
	targets = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0], dtype=float)

	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(units=20, activation="relu", input_shape=(1,)),
		tf.keras.layers.Dense(units=1)
	])

	model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
	model.fit(x=features, y=targets, epochs=500)

	# Return the model here
	return model


# save your model in the .h5 format.

if __name__ == "__main__":
	model = solution_model()
	print(model.predict([7.0]))
	model.save("my_model.h5")

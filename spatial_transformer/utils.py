import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

def img_to_array(data_path, desired_size=None, view=False):
	"""
	Util function for loading RGB image into 4D numpy array.

	Returns array of shape (1, H, W, C)

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	img = Image.open(data_path)
	img = img.convert('RGB')
	if desired_size:
		img = img.resize((desired_size[1], desired_size[0]))
	if view:
		img.show()

	# preprocess	
	x = np.asarray(img, dtype='float32')
	x = np.expand_dims(x, axis=0)
	x /= 255.0

	return x

def array_to_img(x):
	"""
	Util function for converting 4D numpy array to numpy array.

	Returns PIL RGB image.

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	x = np.asarray(x)
	x += max(-np.min(x), 0)
	x_max = np.max(x)
	if x_max != 0:
		x /= x_max
	x *= 255
	return Image.fromarray(x.astype('uint8'), 'RGB')

def run_op(x):
	"""
	Utility function for debugging in tensorflow.

	Runs session to convert tensor to numpy array.
	"""
	# intialize the variable
	init_op = tf.global_variables_initializer()

	# run the graph
	with tf.Session() as sess:
		sess.run(init_op)
		return sess.run(x)

def visualize_grid(Xs, ubound=255.0, padding=1):
	"""
	Reshape a 4D tensor of image data to a grid for easy visualization.

	Inputs:
	- Xs: Data of shape (N, H, W, C)
	- ubound: Output grid will have values scaled to the range [0, ubound]
	- padding: The number of blank pixels between elements of the grid

	Returns:
	- grid

	References:
	- Adapted from CS231n - http://cs231n.github.io/
	"""

	(N, H, W, C) = Xs.shape
	grid_size = int(np.ceil(np.sqrt(N)))
	grid_height = H * grid_size + padding * (grid_size - 1)
	grid_width = W * grid_size + padding * (grid_size - 1)
	grid = np.zeros((grid_height, grid_width, C))
	next_idx = 0
	y0, y1 = 0, H

	for y in range(grid_size):
		x0, x1 = 0, W
		for x in range(grid_size):
			if next_idx < N:
				img = Xs[next_idx]
				low, high = np.min(img), np.max(img)
				grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
				next_idx += 1
			x0 += W + padding
			x1 += W + padding
		y0 += H + padding
		y1 += H + padding
	return grid

def view_images(X, ubound=1.0, save=False, name=''):
	""" Quick helper function to view rgb or gray images."""
	if X.ndim == 3:
		H, W, C = X.shape
		X = X.reshape(H, W, C, 1)
		grid = visualize_grid(X, ubound)
		H, W, C = grid.shape
		grid = grid.reshape((H, W))
		plt.imshow(grid, cmap="Greys_r")
		if save:
			plt.savefig('/Users/kevin/Desktop/' + name, format='png', dpi=1000)
		plt.show()
	elif X.ndim == 4:
		grid = visualize_grid(X, ubound)
		plt.imshow(grid)
		if save:
			plt.savefig('/Users/kevin/Desktop/' + name, format='png', dpi=1000)
		plt.show()
	else:
		raise ValueError
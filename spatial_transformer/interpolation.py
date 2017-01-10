import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img

DIMS = (400, 400)
CAT1 = 'cat1.jpg'
CAT2 = 'cat2.jpg'

def load_data(dims, img_name, view=False):
	"""
	Util function for processing RGB image into 4D tensor.

	Returns tensor of shape (1, H, W, C)
	"""
	image_path = './data/' + img_name
	img = load_img(image_path, target_size=dims)
	if view:
		img.show()
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = img / 255.0
	return img

def affine_grid_generator(height, width, thetas):
	"""
	This function returns a sampling grid, which when
	used with the bilinear sampler on the input feature 
	map, will create an output feature map that is an 
	affine transformation [1] of the input feature map.

	Input
	-----
	- theta: affine transform matrices of shape (num_batch, 2, 3). 
	  For each image in the batch, we have 6 theta parameters of 
	  the form (2x3) that define the affine transformation T.

	Returns
	-------
	- normalized gird (-1, 1) of shape (num_batch, H, W, 2).
	  The 4th dimension has 2 components: (x, y) which are the 
	  sampling points of the original image for each point in the
	  target image.

	Note
	----
	[1]: the affine transformation allows cropping, translation, 
		 and isotropic scaling.
	"""
	# grab batch size
	num_batch = thetas.shape[0]

	# create normalized 2D grid
	x = np.linspace(-1, 1, height)
	y = np.linspace(-1, 1, width)
	x_t, y_t = np.meshgrid(x, y)

	# x_t = K.variable(x_t)
	# y_t = K.variable(y_t)

	# reshape to (xt, yt, 1)
	ones = np.ones(np.prod(x_t.shape))
	sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

	# repeat grid num_batch times
	sampling_grid = np.resize(sampling_grid, (num_batch, 3, height*width))
	
	# transform the sampling grid - batch multiply
	batch_grids = np.matmul(thetas, sampling_grid)
	# batch grid has shape (num_batch, 2, H*W)

	# reshape to (num_batch, H, W, 2)
	batch_grids = batch_grids.reshape(num_batch, 2, height, width)
	batch_grids = np.moveaxis(batch_grids, 1, -1)

	# sanity check
	print("Thetas: {}".format(thetas.shape))
	print("Sampling Grid: {}".format(sampling_grid.shape))
	print("Batch Grids: {}".format(batch_grids.shape))

	return batch_grids

def bilinear_sampler(input_img, x, y):
	"""
	Performs bilinear sampling of the input images according to the 
	normalized coordinates provided by the sampling grid. Note that 
	the sampling is done identically for each channel of the input.

	To test if the function works properly, output image should be
	identical to input image when theta is initialized to identity
	transform.

	Input
	-----
	- input_imgs: batch of images in (B, H, W, C) layout.
	- grid: x, y which is the output of affine_grid_generator.

	Returns
	-------
	- interpolated images according to grids. Same size as grid.

	"""
	# grab dimensions
	B, H, W, C = input_img.shape

	# rescale x and y to [0, W/H]
	x = ((x + 1.) * W) * 0.5
	y = ((y + 1.) * H) * 0.5

	# grab 4 nearest corner points for each (x_i, y_i)
	x0 = np.floor(x).astype(np.int64)
	x1 = x0 + 1
	y0 = np.floor(y).astype(np.int64)
	y1 = y0 + 1

	# make sure it's inside img range [0, H] or [0, W]
	x0 = np.clip(x0, 0, W-1)
	x1 = np.clip(x1, 0, W-1)
	y0 = np.clip(y0, 0, H-1)
	y1 = np.clip(y1, 0, H-1)

	# look up pixel values at corner coords
	Ia = input_img[np.arange(B)[:,None,None], y0, x0]
	Ib = input_img[np.arange(B)[:,None,None], y1, x0]
	Ic = input_img[np.arange(B)[:,None,None], y0, x1]
	Id = input_img[np.arange(B)[:,None,None], y1, x1]

	# calculate deltas
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	# add dimension for addition
	wa = np.expand_dims(wa, axis=3)
	wb = np.expand_dims(wb, axis=3)
	wc = np.expand_dims(wc, axis=3)
	wd = np.expand_dims(wd, axis=3)

	# compute output
	out = wa*Ia + wb*Ib + wc*Ic + wd*Id

	return out

def main():

	# load 4 cat images
	img1 = load_data(DIMS, CAT1, view=True)
	img2 = load_data(DIMS, CAT2)

	# concat into tensor of shape (2, 400, 400, 3)
	input_img = np.concatenate([img1, img2], axis=0)

	# dimension sanity check
	print("Input Img Shape: {}".format(input_img.shape))

	# grab shape
	B, H, W, C = input_img.shape

	# initialize theta to identity transform
	theta = np.array([[0.707, -0.707, 0.], [0.707, 0.707, 0.]])

	# repeat num_batch times
	theta = np.resize(theta, (B, 2, 3))

	# get grids
	batch_grids = affine_grid_generator(H, W, theta)

	x_s = batch_grids[:, :, :, 0:1].squeeze()
	y_s = batch_grids[:, :, :, 1:2].squeeze()

	out = bilinear_sampler(input_img, x_s, y_s)

	img_out = array_to_img(out[0])
	img_out.show()

if __name__ == '__main__':
	main()

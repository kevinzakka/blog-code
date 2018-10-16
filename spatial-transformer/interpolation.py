import numpy as np

from utils import *

def main():

    DIMS = (500, 500)
    data_path = './data/'

    # load 4 cat images
    img1 = img_to_array(data_path + 'cat1.jpg', DIMS)
    img2 = img_to_array(data_path + 'cat2.jpg', DIMS, view=True)

    # concat into tensor of shape (2, 400, 400, 3)
    input_img = np.concatenate([img1, img2], axis=0)

    # dimension sanity check
    print("Input Img Shape: {}".format(input_img.shape))

    # grab shape
    B, H, W, C = input_img.shape

    # initialize theta to identity transform
    M = np.array([[1., 0., 0.], [0., 1., 0.]])

    # repeat num_batch times
    M = np.resize(M, (B, 2, 3))

    # get grids
    batch_grids = affine_grid_generator(H, W, M)

    x_s = batch_grids[:, :, :, 0:1].squeeze()
    y_s = batch_grids[:, :, :, 1:2].squeeze()

    out = bilinear_sampler(input_img, x_s, y_s)
    print("Out Img Shape: {}".format(out.shape))

    # view the 2nd image
    array_to_img(out[-1]).show()


if __name__ == '__main__':
    main()

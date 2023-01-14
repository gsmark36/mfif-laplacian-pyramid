import os.path as path
import skimage.io as io
import numpy as np
from skimage import color
from skimage import util
import matplotlib.pyplot as plt
import time

from LP_fusion_rgb import laplacian_pyramid
from LP_fusion_rgb import pyramid_fusion
from LP_fusion_rgb import get_entropy

def top_average(img_1, img_2):
    img_output = np.zeros(np.shape(img_1))
    
    for y in range(0, img_1.shape[0]):
        for x in range(0, img_1.shape[1]):
            for cc in range(img_1.shape[2]):
                img_output[y, x, cc] = (img_1[y, x, cc] + img_2[y, x, cc])/2

    return np.squeeze(img_output)


def other_max(img_1, img_2):
    img_output = np.zeros(np.shape(img_1))
    
    for y in range(0, img_1.shape[0]):
        for x in range(0, img_1.shape[1]):
            for cc in range(img_1.shape[2]):
                if img_1[y, x, cc] >= img_2[y, x, cc]:
                    img_output[y, x, cc] = img_1[y, x, cc]
                elif img_1[y, x, cc] < img_2[y, x, cc]:
                    img_output[y, x, cc] = img_2[y, x, cc]

    return np.squeeze(img_output)


def pyramid_average(pyr_1, pyr_2, gray = False):
    pyr_output = []
    levels = np.size(pyr_1)

    # Apply fusion operations to get fused Laplacian pyramid
    for i in range(0, levels):
        temp_1 = pyr_1[i]
        temp_2 = pyr_2[i]

        # Reshape each layer if grayscale is required
        if gray == True:
            temp_1 = temp_1.reshape(temp_1.shape[0], temp_1.shape[1], -1)
            temp_2 = temp_2.reshape(temp_2.shape[0], temp_2.shape[1], -1)

        if i == levels-1:
            pyr_output.append(top_average(temp_1, temp_2))
        else:
            pyr_output.append(other_max(temp_1, temp_2))
    
    if np.size(pyr_output) != levels:
        print('Error: Image fusion failed.')
        return None
    
    return pyr_output


def RMSE(ground, result):
    ground = ground.reshape(ground.shape[0], ground.shape[1], -1)
    result = result.reshape(result.shape[0], result.shape[1], -1)
    squared_error = 0

    for y in range(0, ground.shape[0]):
        for x in range(0, ground.shape[1]):
            for cc in range(ground.shape[2]):
                squared_error += (ground[y, x, cc] - result[y, x, cc])**2
    
    output = np.sqrt(squared_error/(ground.shape[0]*ground.shape[1]))
    return output
    

if __name__ == "__main__":
    img_0 = io.imread(path.join('Images','cc0.png'))
    img_1 = io.imread(path.join('Images','cc1.png'))
    img_2 = io.imread(path.join('Images','cc2.png'))

    # Check sizes of input images
    if np.shape(img_1) != np.shape(img_2):
        print('Error: Image sizes do not match.')
        exit()
    
    # Set the maximum number of Laplacian pyramid levels
    # Default value = (-1) = maximum possible number of levels
    max_levels = 6

    # Choose RGB or grayscale
    grayscale = False
    
    # Enable or disable ground truth
    groundtruth = True

    if np.size(np.shape(img_1)) == 3:
        img_1 = util.img_as_float(img_1[:,:,:3])
        img_2 = util.img_as_float(img_2[:,:,:3])
        if groundtruth == True:
            img_0 = util.img_as_float(img_0[:,:,:3])
        if grayscale == True:
            img_1 = color.rgb2gray(img_1)
            img_2 = color.rgb2gray(img_2)
            if groundtruth == True:
                img_0 = color.rgb2gray(img_2)
    elif np.size(np.shape(img_1)) == 2:
        img_1 = util.img_as_float(img_1[:,:])
        img_2 = util.img_as_float(img_2[:,:])
        if groundtruth == True:
            img_0 = util.img_as_float(img_0[:,:])
        print('Input images are in grayscale.')
        grayscale = True
    else:
        print('Error: Image sizes are invalid.')
        exit()

    pyramid_1 = laplacian_pyramid.decompose(img_1, levels=max_levels)
    pyramid_2 = laplacian_pyramid.decompose(img_2, levels=max_levels)

    start_time = time.time()
    pyramid_f = pyramid_fusion(pyramid_1, pyramid_2, gray=grayscale)
    img_f = laplacian_pyramid.reconstruct(pyramid_f)
    end_time = time.time()
    print('Time for fusion =', end_time-start_time, 's')

    start_time = time.time()
    pyramid_a = pyramid_average(pyramid_1, pyramid_2, gray=grayscale)
    img_a = laplacian_pyramid.reconstruct(pyramid_a)
    end_time = time.time()
    print('Time for average method =', end_time-start_time, 's')

    std_f = np.std(img_f)
    std_a = np.std(img_a)
    h_f = get_entropy(img_f)
    h_a = get_entropy(img_a)
    
    print('STD for fusion =', std_f)
    print('STD for avearge method =', std_a)
    print('Entropy for fusion =', h_f)
    print('Entropy for average method =', h_a)

    if groundtruth == True:
        rmse_f = RMSE(img_0, img_f)
        rmse_a = RMSE(img_0, img_a)
        print('RMSE for fusion =', rmse_f)
        print('RMSE for average method =', rmse_a)

    plt.figure()
    if grayscale == True:
        plt.imshow(img_f, cmap='gray')
        #plt.imsave('result_grayscale.png', np.clip(img_f, 0, 1), cmap='gray')
    else:
        plt.imshow(np.clip(img_f, 0, 1))
        #plt.imsave('result_rgb.png', np.clip(img_f, 0, 1))
    plt.show()

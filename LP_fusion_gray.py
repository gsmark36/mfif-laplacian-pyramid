import os.path as path
import skimage.io as io
import math
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import matplotlib.pyplot as plt
import time

# Fusion of two images (grayscale)

def gausspyr_reduce(x, kernel_a=0.4):
    # Kernel
    K = np.array([0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2])
    
    x = x.reshape(x.shape[0], x.shape[1], -1) # Add an extra dimension if grayscale
    y = np.zeros([math.ceil(x.shape[0]/2), math.ceil(x.shape[1]/2), x.shape[2]]) # Store the result in this array
    
    for cc in range(x.shape[2]): # for each colour channel
        # Step 1: filter rows
        y_a = sp.signal.convolve2d(x[:,:,cc], K.reshape(1,-1), mode='same', boundary='symm')
        # Step 2: subsample rows (skip every second column)
        y_a = x[:,::2,cc]
        # Step 3: filter columns
        y_a = sp.signal.convolve2d(y_a, K.reshape(-1,1), mode='same', boundary='symm')
        # Step 4: subsample columns (skip every second row)
        y[:,:,cc] = y_a[::2,:]
    return np.squeeze(y) # remove an extra dimension for grayscale images
  

def gausspyr_expand(x, sz=None, kernel_a=0.4):
    # Kernel is multipled by 2 to preserve energy when increasing the resolution
    K = 2*np.array([0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2])
  
    # Size of the output image
    if sz is None:
      sz = (x.shape[0]*2, x.shape[1]*2)
    
    x = x.reshape(x.shape[0], x.shape[1], -1) # Add an extra dimension if grayscale
    y = np.zeros([sz[0], sz[1], x.shape[2]]) # Store the result in this array
  
    for cc in range(x.shape[2]): # for each colour channel
        y_a = np.zeros((x.shape[0], sz[1]))
        # Step 1: upsample rows
        y_a[:,::2] = x[:,:,cc]
        # Step 2: filter rows
        y_a = sp.signal.convolve2d(y_a, K.reshape(1,-1), mode='same', boundary='symm')
        # Step 3: upsample columns
        y[::2,:,cc] = y_a
        # Step 4: filter columns
        y[:,:,cc] = sp.signal.convolve2d(y[:,:,cc], K.reshape(-1,1), mode='same', boundary='symm')
    return np.squeeze(y) # remove an extra dimension for grayscale images


class laplacian_pyramid:

    @staticmethod
    def decompose(img, levels=-1):    
        """
        Decompose img into a Laplacian pyramid. 
        levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created. 
        """
        # The maximum number of levels
        max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))
  
        assert levels < max_levels
  
        if levels == -1: 
            levels = max_levels
  
        pyramid = []
        gausspyr = []

        # Find each layer of Laplacian pyramid
        for i in range(1, levels+1):
            if i == 1:
                gausspyr.append(img)
            else:
                gausspyr.append(gausspyr_reduce(gausspyr[i-2]))
                pyramid.append(gausspyr[i-2] - gausspyr_expand(gausspyr[i-1], sz=np.shape(gausspyr[i-2])))

        pyramid.append(gausspyr[levels-1])
        return pyramid
  
    @staticmethod
    def reconstruct(pyramid):    
        """
        Combine the levels of the Laplacian pyramid to reconstruct an image. 
        """
        img = None
        levels = np.size(pyramid)
        img = pyramid[levels-1]
        
        # Perform inverse operation to reconstruct the original image
        for i in range(1, levels):
            img = gausspyr_expand(img, sz=np.shape(pyramid[levels-i-1])) + pyramid[levels-i-1]
        return img


def get_entropy(img):
    vec = img.flatten()
    total_pix = len(vec)
    prob = {}

    # Find the probability of occurence for each pixel value
    for pix in vec:
        if pix in prob:
            prob[pix] += 1
        else:
            prob[pix] = 1

    for pix, count in prob.items():
        prob[pix] /= total_pix

    # Calculate the entropy of input image
    entropy = 0
    for pix, probability in prob.items():
        entropy -= probability * np.log2(probability)
    return entropy


def get_energy(img):
    vec = img.flatten()
    energy = 0

    # Calculate the regional energy of input image
    for pix in range(0, len(vec)):
        energy += vec[pix]**2
    return energy


def top_fusion(img_1, img_2, sz_h, sz_w):
    img_output = np.zeros(np.shape(img_1))
    
    h = sz_h
    w = sz_w
    ext_x = int(w / 2)
    ext_y = int(h / 2)
    
    # Extend input images with respect to the size of sliding window
    edge_x = np.zeros([img_1.shape[0], ext_x], img_1.dtype)
    tem_img_1 = np.hstack((edge_x, img_1, edge_x))
    edge_y = np.zeros([ext_y, tem_img_1.shape[1]], img_1.dtype)
    ext_img_1 = np.vstack((edge_y, tem_img_1, edge_y))
    tem_img_2 = np.hstack((edge_x, img_2, edge_x))
    ext_img_2 = np.vstack((edge_y, tem_img_2, edge_y))

    for y in range(0, img_1.shape[0]):
        for x in range(0, img_1.shape[1]):
            w_1 = ext_img_1[y:y+2*ext_y+1, x:x+2*ext_x+1]
            w_2 = ext_img_2[y:y+2*ext_y+1, x:x+2*ext_x+1]
            # Deviation or Variance
            D1 = np.std(w_1)**2
            D2 = np.std(w_2)**2
            # Entropy
            H1 = get_entropy(w_1)
            H2 = get_entropy(w_2)
            # Fusion strategy based on calculated coefficients
            if D1 >= D2 and H1 >= H2:
                img_output[y, x] = img_1[y, x]
            elif D1 < D2 and H1 < H2:
                img_output[y, x] = img_2[y, x]
            else:
                img_output[y, x] = (img_1[y, x] + img_2[y, x])/2

    return img_output


def other_fusion(img_1, img_2, sz_h, sz_w):
    img_output = np.zeros(np.shape(img_1))
    
    h = sz_h
    w = sz_w
    ext_x = int(w / 2)
    ext_y = int(h / 2)

    # Extend input images with respect to the size of sliding window
    edge_x = np.zeros([img_1.shape[0], ext_x], img_1.dtype)
    tem_img_1 = np.hstack((edge_x, img_1, edge_x))
    edge_y = np.zeros([ext_y, tem_img_1.shape[1]], img_1.dtype)
    ext_img_1 = np.vstack((edge_y, tem_img_1, edge_y))
    tem_img_2 = np.hstack((edge_x, img_2, edge_x))
    ext_img_2 = np.vstack((edge_y, tem_img_2, edge_y))

    for y in range(0, img_1.shape[0]):
        for x in range(0, img_1.shape[1]):
            w_1 = ext_img_1[y:y+2*ext_y+1, x:x+2*ext_x+1]
            w_2 = ext_img_2[y:y+2*ext_y+1, x:x+2*ext_x+1]
            # Regional energy
            RE1 = abs(get_energy(w_1))
            RE2 = abs(get_energy(w_2))
            # Fusion strategy
            if RE1 >= RE2:
                img_output[y, x] = img_1[y, x]
            elif RE1 < RE2:
                img_output[y, x] = img_2[y, x]

    return img_output


def pyramid_fusion(pyr_1, pyr_2):
    pyr_output = []
    levels = np.size(pyr_1)

    # Apply fusion operations to get fused Laplacian pyramid
    for i in range(0, levels):
        temp_1 = pyr_1[i]
        temp_2 = pyr_2[i]
        if i == levels-1:
            pyr_output.append(top_fusion(temp_1, temp_2, sz_h=3, sz_w=3))
        else:
            pyr_output.append(other_fusion(temp_1, temp_2, sz_h=3, sz_w=3))
    
    if np.size(pyr_output) != levels:
        print('Error: Image fusion failed.')
        return None
    
    return pyr_output


if __name__ == "__main__":
    img_1 = io.imread(path.join('Images','kh1.png'))
    img_2 = io.imread(path.join('Images','kh2.png'))
    
    # Check sizes of input images
    if np.shape(img_1) != np.shape(img_2):
        print('Error: Image sizes do not match.')
        exit()
    
    # Check RGB or grayscale
    if np.size(np.shape(img_1)) == 3:
        img_1 = util.img_as_float(img_1[:,:,:3])
        img_2 = util.img_as_float(img_2[:,:,:3])
        img_1 = color.rgb2gray(img_1)
        img_2 = color.rgb2gray(img_2)
    elif np.size(np.shape(img_1)) == 2:
        img_1 = util.img_as_float(img_1[:,:])
        img_2 = util.img_as_float(img_2[:,:])
        print('Input images are in grayscale.')
    else:
        print('Error: Image sizes are invalid.')
        exit()

    # Set the maximum number of Laplacian pyramid levels
    # Default value = (-1) = maximum possible number of levels
    max_levels = 6

    # Initialize time counter
    start_time = time.time()

    pyramid_1 = laplacian_pyramid.decompose(img_1, levels=max_levels)
    pyramid_2 = laplacian_pyramid.decompose(img_2, levels=max_levels)
    pyramid_f = pyramid_fusion(pyramid_1, pyramid_2)

    img_f = laplacian_pyramid.reconstruct(pyramid_f)
    
    end_time = time.time()
    print('Processing time =', end_time-start_time, 's')

    # Visualize fused Laplacian pyramid
    plt.figure(figsize=(3*len(pyramid_f), 3))
    grid = len(pyramid_f) * 10 + 101
    for i, layer in enumerate(pyramid_f):
        plt.subplot(grid+i)
        plt.title('Level {}'.format(i))
        plt.axis('off')
        if i == len(pyramid_f)-1:
            io.imshow(layer)
        else:
            plt.imshow(layer)
    plt.show()

    # Show final image
    plt.figure()
    plt.imshow(img_f, cmap='gray')
    plt.show()

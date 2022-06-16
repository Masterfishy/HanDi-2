"""
This file contains the methods used to calculate the Histogram of Oriented Gradients (HOG) for a given image.
"""

import numpy as np

def hog_horizontal_gradient(image):
    """
    Calculate the horizontal gradient of a given image.

    :param image: ndarray of pixel values
    :type image: ndarray
    :return: ndarray of gradient values for each pixel
    :rtype: ndarray
    """
    n_row, n_col = image.shape
    result = np.zeros(image.shape)

    for r in range(n_row):
        for c in range(1, n_col - 1):
            result[r, c] = image[r, c + 1] - image[r, c - 1]

    return result

def hog_vertical_gradient(image):
    """
    Calculate the vertical gradient of a given image.

    :param image: ndarray of pixel values
    :type image: ndarray
    :return: ndarray of gradient values for each pixel
    :rtype: ndarray
    """
    n_row, n_col = image.shape
    result = np.zeros(image.shape)

    for c in range(n_col):
        for r in range(1, n_row - 1):
            result[r, c] = image[r + 1, c] - image[r - 1, c]

    return result

def hog_gradient_magnitudes(horizontal_gradient, vertical_gradient):
    """
    Calculate the magnitude of pixel values based on horizontal and vertical gradients.

    :param horizontal_gradient: ndarray of gradient values calculated along the x axis
    :type horizontal_gradient: ndarray
    :param vertical_gradient: ndarray of gradient values calculated along the y axis
    :type vertical_gradient: ndarray
    :return: ndarray of magnitudes for each pixel.
    :rtype: ndarray
    """
    n_row, n_col = horizontal_gradient.shape
    result = np.zeros(horizontal_gradient.shape)

    for r in range(n_row):
        for c in range(n_col):
            value = np.power(horizontal_gradient[r, c], 2) + np.power(vertical_gradient[r, c], 2)
            result[r, c] = np.sqrt(value)

    return result

def hog_gradient_directions(horizontal_gradient, vertical_gradient):
    """
    Calculate the gradient direction of pixel values based on horizontal and vertical gradients.

    :param horizontal_gradient: ndarray of gradient values calculated along the x axis
    :type horizontal_gradient: ndarray
    :param vertical_gradient: ndarray of gradient values calculated along the y axis
    :type vertical_gradient: ndarray
    :return: ndarray of directions for each pixel.
    :rtype: ndarray
    """
    n_row, n_col = horizontal_gradient.shape
    result = np.zeros(horizontal_gradient.shape)

    for r in range(n_row):
        for c in range(n_col):
            angle = np.arctan(vertical_gradient[r, c] / (horizontal_gradient[r, c] + 1e-5))
            result[r, c] = np.abs(np.rad2deg(angle))

    return result

def hog_histogram(magnitudes, directions, bins):
    hist = np.zeros(bins)
    n_row, n_col = magnitudes.shape
    step = 180 // bins

    for r in range(n_row):
        for c in range(n_col):
            bin = int(directions[r, c]) // step
            hist[bin] += magnitudes[r, c]

    return hist

def hog(magnitudes, directions, bins=9, window=4):
    
    n_row, n_col = magnitudes.shape
    nr = n_row - window
    nc = n_col - window
    hist = np.zeros((nr * nc, bins))
    counter = 0

    for r in range(nr):
        for c in range(nc):
            window_mags = magnitudes[r:r+window, c:c+window]
            window_dirs = directions[r:r+window, c:c+window]
            hist[counter] = hog_histogram(window_mags, window_dirs, bins)
            counter += 1

    return hist


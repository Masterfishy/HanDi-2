"""
This file contains the methods used to calculate the Histogram of Oriented Gradients (HOG) for a given image.
"""
import numpy as np

def hog_horizontal_gradient(image, shape):
    """
    Calculate the horizontal gradient of a given image.

    :param image: 1 dimensional vector of feature values
    :type image: ndarray
    :param shape: the dimensions of the image
    :type shape: tuple
    :return: ndarray of gradient values for each feature
    :rtype: ndarray
    """
    n_row, n_col = shape
    result = np.zeros(image.shape)
    index = 0

    for r in range(n_row):
        for c in range(1, n_col - 1):
            result[r + c + index] = image[r + c + index + 1] - image[r + c + index - 1]
        index += n_col - 1

    return result

def hog_vertical_gradient(image):
    """
    Calculate the vertical gradient of a given image.

    :param image: ndarray of feature values
    :type image: ndarray
    :return: ndarray of gradient values for each feature
    :rtype: ndarray
    """
    n_row, n_col = image.shape
    result = np.zeros(image.shape)

    for r in range(1, n_row - 1):
        for c in range(n_col):
            result[r, c] = image[r + 1, c] - image[r - 1, c]

    return result

def hog_gradient_magnitudes(horizontal_gradient, vertical_gradient):
    """
    Calculate the magnitude of feature values based on horizontal and vertical gradients.

    :param horizontal_gradient: ndarray of gradient values calculated along the x axis
    :type horizontal_gradient: ndarray
    :param vertical_gradient: ndarray of gradient values calculated along the y axis
    :type vertical_gradient: ndarray
    :return: ndarray of magnitudes for each feature
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
    Calculate the gradient direction of feature values based on horizontal and vertical gradients.

    :param horizontal_gradient: ndarray of gradient values calculated along the x axis
    :type horizontal_gradient: ndarray
    :param vertical_gradient: ndarray of gradient values calculated along the y axis
    :type vertical_gradient: ndarray
    :return: ndarray of directions in degrees for each feature.
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
    """
    Calculates the frequencies of the given directions.

    :param magnitudes: ndarray of magnitudes
    :type magnitudes: ndarray
    :param directions: ndarray of directions in degrees
    :type directions: ndarray
    :param bins: the number of bins to chunk 180 degrees
    :type bins: int
    :return: one dimensional array of frequencies for each bin
    :rtype: ndarray
    """
    hist = np.zeros(bins)
    n_row, n_col = magnitudes.shape
    step = 180 // bins

    for r in range(n_row):
        for c in range(n_col):
            dir = directions[r, c]
            mag = magnitudes[r, c]

            lower_bin = int(np.floor(dir / step))
            upper_bin = int(np.ceil(dir / step))

            lower_dir = lower_bin * step
            upper_dir = upper_bin * step

            hist[lower_bin] += mag * (1 - (np.abs(dir - lower_dir) / step))
            hist[upper_bin] += mag * (1 - (np.abs(dir - upper_dir) / step))

    return hist

def hog_image_features(magnitudes, directions, bins=9, window=4):
    """
    Calculate the frequencies of the given directions in a sliding window.
    
    :param magnitudes: ndarray of magnitudes
    :type magnitudes: ndarray
    :param directions: ndarray of directions in degrees
    :type directions: ndarray
    :param bins: the number of bins to chunk 180 degrees
    :type bins: int
    :param window: the size of the sliding square window (window x window)
    :type window: int
    :return: ndarray of histograms for each window
    :rtype: ndarray
    """
    n_row, n_col = magnitudes.shape
    nr = n_row - window
    nc = n_col - window
    hists = np.zeros((nr * nc, bins))
    counter = 0

    for r in range(nr):
        for c in range(nc):
            window_mags = magnitudes[r:r+window, c:c+window]
            window_dirs = directions[r:r+window, c:c+window]
            hists[counter] = hog_histogram(window_mags, window_dirs, bins)
            counter += 1

    return hists


def hog(image, bins=9, window=4):
    """
    Calculate the Histogram of Oriented Gradients for the given image.

    :param image: ndarray of feature values
    :type image: ndarray
    :param bins: the number of bins to chunk 180 degrees
    :type bins: int
    :param window: the size of the sliding square window (window x window)
    :type window: int
    :return: feature values for each window
    :rtype: ndarray
    """
    horz_grad = hog_horizontal_gradient(image)
    vert_grad = hog_vertical_gradient(image)

    mags = hog_gradient_magnitudes(horz_grad, vert_grad)
    dirs = hog_gradient_directions(horz_grad, vert_grad)

    features = hog_image_features(mags, dirs, bins, window)

    return features

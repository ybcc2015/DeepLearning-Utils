import numpy as np
import cv2
from imutils import paths


def calculate_mean_std(img_root, channels=3):
    """
    Calculate the mean and standard deviation of the training images.

    Arguments:
        img_root {str} -- the root directory of training images
        channels {int} -- the numbers of channles

    Returns:
        mean {1-d numpy array} -- mean value of each channel
        std {1-d numpy array} -- standard deviation  of each channel
    """
    total_pixel = 0
    channel_sum = np.zeros(channels)
    channel_square_sum = np.zeros(channels)

    for img_path in paths.list_images(img_root):
        img = cv2.imread(img_path)
        img = img / 255.
        channel_sum = np.sum(img, axis=(0, 1))
        channel_square_sum = np.sum(img ** 2, axis=(0, 1))
        total_pixel += img.shape[0] * img.shape[1]

    mean = channel_sum / total_pixel
    std = np.sqrt(channel_square_sum / total_pixel - mean ** 2)

    if channels == 3:  # bgr -> rgb
        mean = mean[::-1]
        std = std[::-1]

    return mean, std

import cv2
import numpy as np
from skimage import feature


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius)

        (hist, _) = np.histogram(lbp.flatten(),
            bins = np.arange(0, 256),
            range=(0, 255))
        return hist
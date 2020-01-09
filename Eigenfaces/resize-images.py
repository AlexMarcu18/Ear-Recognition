import cv2
import os

WIDTH = 100
HEIGHT = 100
dim = (WIDTH, HEIGHT)

dirName = '../TrainingData'
imagesPaths = os.listdir(dirName)

def resizeImage(dirName, path):
    image = cv2.imread(os.path.join(dirName, path))
    resizedImage = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite('../ResizedTrainingData/' + path, resizedImage)

for path in imagesPaths:
    resizeImage(dirName, path)
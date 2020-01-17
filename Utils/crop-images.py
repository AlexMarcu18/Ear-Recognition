import cv2
import os


# dirName = '../TrainingData'
# imagesPaths = os.listdir(dirName)
# img = cv2.imread(os.path.join(dirName, imagesPaths[1]))
# cv2.imshow('Res', img[40:630, 35:465])
# cv2.waitKey(0)
dirName = '../TrainingDataSet2'
imagesPaths = os.listdir(dirName)

def resizeImage(dirName, path):
    image = cv2.imread(os.path.join(dirName, path))
    resizedImage = image[40:630, 35:465]
    cv2.imwrite('../CroppedTrainingDataSet2/' + path, resizedImage)

for path in imagesPaths:
    resizeImage(dirName, path)
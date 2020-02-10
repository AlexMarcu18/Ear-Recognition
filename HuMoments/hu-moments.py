import cv2
import os
import math


def euclideanDistance(A, B):
    suma = 0
    for i in range(0,7):
        suma += pow(B[i] - A[i], 2)
    return math.sqrt(suma)

def logNormalize(moment):
    if moment:
        return -1* math.copysign(1.0, moment) * math.log10(abs(moment))
    return 0

huMomentsMatrix = []
dirName = '../TrainingData-CVLOO'
imagesPaths = os.listdir(dirName)
imagesPaths.sort()
print('Calculating Hu Moments for our dataset..')
for path in imagesPaths:
    im = cv2.imread(os.path.join(dirName, path), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('res', im)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 1)
    # cv2.imshow('rest', im)
    # cv2.waitKey(0)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    result = map(logNormalize, huMoments)
    huMomentsMatrix.append((list(result), path.split('_')[0]))

dirName = '../TestData-CVLOO'
imagesPaths = os.listdir(dirName)
imagesPaths.sort()

print('Starting the tests..')
recognised = 0
for path in imagesPaths:
    im = cv2.imread(os.path.join(dirName, path), cv2.IMREAD_GRAYSCALE)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 1)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    result = map(logNormalize, huMoments)
    testIm = (list(result), path.split('_')[0])

    distMatrix = []
    for i in huMomentsMatrix:
        dist = euclideanDistance(testIm[0], i[0])
        distMatrix.append((dist, i[1]))

    distMatrix.sort()
    if distMatrix[0][1] == testIm[1]:
        # print(distMatrix[0][1], testIm[1])
        recognised += 1

percentage = recognised/50 * 100
print(percentage)
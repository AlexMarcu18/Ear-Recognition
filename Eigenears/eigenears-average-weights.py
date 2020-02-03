import cv2
import numpy as np
import os
import sys

# np.set_printoptions(threshold=sys.maxsize)
# Number of EigenFaces
NUM_EIGEN_FACES = 10
# Maximum weight
MAX_SLIDER_VALUE = 255
dirName = '../ResizedTrainingData'

def readImages(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    return image

def createDataMatrix(images, size):
    print('Creating data matrix..',end=" ... ")
    numImages = len(images)
    data = np.zeros((numImages, size[0] * size[1] * size[2]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        data[i,:] = image
    print("Done")
    return data

def reshapeVectorToFaces(eigenvectors, size):
    for i in range(0, NUM_EIGEN_FACES):
        eigenFace = eigenvectors[i].reshape(size)
        eigenFaces.append(eigenFace)

def getWeights(substractedImageData, eigenvectors, flag):
    normalizedEigenvectors = []

    # NORMALIZE EIGENVECTORS
    for i in range(0, NUM_EIGEN_FACES):
        normalizedEigenvectors.append(cv2.normalize(eigenvectors[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8).reshape(size[0] * size[1] * size[2]))
    
    # CALCULATE WEIGHTS
    if flag == 1:
        trainingSetWeightsVector = []
        for i in range(0, len(substractedImageData)):
            weightsVector = []
            for j in range(0, NUM_EIGEN_FACES):
                weightsVector.append(normalizedEigenvectors[j] @ substractedImageData[i])
            trainingSetWeightsVector.append(weightsVector)
        return np.asarray(trainingSetWeightsVector)
    elif flag == 2:
        testSetWeightsVector = []
        for i in range(0, NUM_EIGEN_FACES):
            testSetWeightsVector.append(normalizedEigenvectors[i] @ substractedImageData[0])
        return np.asarray(testSetWeightsVector)
    else: 
        return
    

def substractMeanTrainingSet(dataMatrix, mean):
    for i in range(0, 300):
        imagesDataMinusMean.append((dataMatrix[i].astype(np.uint8) - mean[0].astype(np.uint8)))

def substractMeanTestSet(dataImage, mean):
    testImageDataMinusMean.append(dataImage.astype(np.uint8) - mean[0].astype(np.uint8))


def showEigenFaces(eigenfaces):
    for i in range(0, NUM_EIGEN_FACES):
        normalizedFace = cv2.normalize(eigenfaces[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
        cv2.imshow('Eigenface no. %s'%(i+1), cv2.resize(normalizedFace, (0,0), fx=2, fy=2))

def showAverageFace(averageface):
    convertedAverageFace = averageFace.astype(np.uint8)
    cv2.imshow("Result", cv2.resize(convertedAverageFace, (0,0), fx=2, fy=2))


# GET IMAGES AND STORE THEM
imagesPaths = os.listdir(dirName)
imagesPaths.sort()
imagesData = [readImages(os.path.join(dirName, path)) for path in imagesPaths]
size = imagesData[0].shape

# TEST IF THE FIRST PHOTO IS STORED CORRECTLY
# firstImage = dataMatrix[0].astype('uint8').reshape(size)
# cv2.imshow('Test', firstImage)

# CREATE IMAGE DATA MATRIX
dataMatrix = createDataMatrix(imagesData, size)

# APPLY PCA ON THE TRAINING SET
print('Data matrix: ', dataMatrix)
print('Calculating PCA...')
mean, eigenvectors = cv2.PCACompute(dataMatrix, mean=None, maxComponents=NUM_EIGEN_FACES)
print('Done')
# print('Mean is:', mean)
# print('EigenVectors are: ', eigenvectors)

# RESHAPE MEAN AND VECTORS TO BE SHOWNABLE
eigenFaces = []
reshapeVectorToFaces(eigenvectors, size)
averageFace = mean.reshape(size)

# SUBSTRACT MEAN FROM TRAINING SET
imagesDataMinusMean = []
substractMeanTrainingSet(dataMatrix, mean)
# cv2.imshow('test', cv2.resize(imagesDataMinusMean[4].astype(np.uint8).reshape(size), (0,0), fx=2, fy=2))

# GET TRAINING SET WEIGHTS
trainingSetWeights = getWeights(imagesDataMinusMean, eigenvectors, 1)
averagePersonWeigths = []

# CALCULATE AVERAGE WEIGHTS FOR EVERY HUMAN IN THE SET
for i in range(0, 50):
    shape = trainingSetWeights[i].shape
    lowerBound = i * 6
    upperBound = (i+1) * 6
    vectorWeightSum = []
    for j in range(lowerBound, upperBound):
        vectorWeightSum.append(trainingSetWeights[j])
    vectorWeightSum = np.asarray(vectorWeightSum, dtype=np.uint8)
    averagePersonWeigths.append((vectorWeightSum[0:len(vectorWeightSum)].sum(0) // 6).astype(np.uint8))


minEuclideanDist = []

testDirName = '../TestData'
imagesNames = os.listdir(testDirName)
imagesNames.sort()

NN = 5

for i in range(0, 50):
    euclideanDist = []
    # LOAD TEST IMAGE
    testImagePath = os.path.join(testDirName, imagesNames[i])
    testImage = cv2.resize(cv2.imread(testImagePath), (size[0], size[1]))
    testData = np.zeros(size[0] * size[1] * size[2])
    testImage = testImage.flatten()

    # SUBSTRACT MEAN FROM TEST SET
    testImageDataMinusMean = []
    substractMeanTestSet(testImage, mean)

    # GET TEST SET WEIGHTS
    testSetWeights = getWeights(testImageDataMinusMean, eigenvectors, 2)

    # CALCULATE EUCLIDEAN DISTANCES
    for i in range(0, 50):
        dist = cv2.norm(averagePersonWeigths[i], testSetWeights)
        euclideanDist.append((dist, i))

    # SAVE MIN DISTANCE
    if NN == 1:
        minEuclideanDist.append(min(euclideanDist))
    elif NN > 1:
        euclideanDist.sort()
        minDists = []
        for i in range(0, NN):
            minDists.append(euclideanDist[i])
        minEuclideanDist.append(minDists)

recognised = 0
if NN == 1:
    for i in range(0, 50):
        if minEuclideanDist[i][1] == i:
            print(i, minEuclideanDist[i][1], minEuclideanDist[i][0])
            recognised += 1
            print('No.%s Recognised'%i)
        else: 
            print('Who are you no %s'%i)
            print(i, minEuclideanDist[i][1], minEuclideanDist[i][0])
        print('Next person \n')
elif NN > 1:
    for i in range(0, 50):
        for j in range(0, NN):
            if minEuclideanDist[i][j][1] == i:
                print('No.%s Recognised'%i)
                recognised += 1
                print(i, minEuclideanDist[i][j][1], minEuclideanDist[i][j][0])
            else: 
                print('Who are you no %s'%i)
                print(i, minEuclideanDist[i][j][1], minEuclideanDist[i][j][0])
        print('Next Person \n')


# testImage = cv2.resize(cv2.imread(testImagePath), (100, 100))
# cv2.imshow('Test', testImage)
percentage = recognised/50 * 100
print(f'The percentage of recognition is: {percentage}%')

# firstImage = dataMatrix[min(errors)[1]].astype('uint8').reshape(size)
# cv2.imshow('Recognized', firstImage)


dirName = '../ResizedTrainingData'

# os.path.join(dirName, images)

# SHOW RESULTS
# showAverageFace(averageFace)
# showEigenFaces(eigenFaces)

cv2.waitKey(0)
cv2.destroyAllWindows()
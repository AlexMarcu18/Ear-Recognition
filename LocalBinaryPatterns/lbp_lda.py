import numpy as np
import cv2
import os
from lbp_algorithm import LocalBinaryPatterns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def calculateHist(imagePath):
    print(f'Investigating {imagePath}')
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = lbp.describe(gray)
    namePath = imagePath.split(os.path.sep)[3]
    label = namePath.split('_')[0]
    labels.append(label)
    data.append(hist)

lbp = LocalBinaryPatterns(10, 10)
data = []
labels = []

dirName = '../Dataset-CVLOO/CroppedTrainingDataSet1'
imagesPaths = os.listdir(dirName)
imagesPaths.sort()

print('Calculating Local Binary Pattern Algorithm using training data')
for path in imagesPaths:
    calculateHist(os.path.join(dirName, path))

lda = LinearDiscriminantAnalysis()
lda.fit(data, labels)

dirName = "../Dataset-CVLOO/TestDataSet1"
imagesPaths = os.listdir(dirName)
imagesPaths.sort()

predictions = []
for imagePath in imagesPaths:
    print(f'Verifying {imagePath}')

    # load and cliasify the test images
    path = os.path.join(dirName,  imagePath)
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = lbp.describe(gray)
    prediction = lda.predict(hist.reshape(1, -1))

    # save the prediction for each image
    label = imagePath.split('_')[0]
    predictions.append((label, prediction[0]))
    # cv2.waitKey(0)

recognised = 0
# see results
for predict in predictions:
    if predict[0] == predict[1]:
        recognised += 1
    print(f"I classifed {predict[0]} as {predict[1]}")

percentage = recognised/50 * 100
print(percentage)

from lbp_algorithm import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import argparse
import cv2
import os
import matplotlib.pyplot as plt

# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--training", required=True,
#     help="path to the training images")
# ap.add_argument("-e", "--testing", required=True,
#     help="path to the testing images")
# args = vars(ap.parse_args())

def calculateHist(imagePath):
    print(f'Investigating {imagePath}')
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = lbp.describe(gray)
    labels.append(imagePath.split(os.path.sep)[2])
    data.append(hist)

lbp = LocalBinaryPatterns(10, 18)
data = []
labels = []


dirName = '../CroppedTrainingDataSet1'
imagesPaths = os.listdir(dirName)
imagesPaths.sort()
personIndex = 0
index = 0
print('Calculating Local Binary Pattern Algorithm using training data')
for path in imagesPaths:
    calculateHist(os.path.join(dirName, path))

print('Training our algorithm based on our histograms')
model = LinearSVC(C=100, max_iter=100000)
model.fit(data, labels)

dirName = "../TestDataSet1"
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
    prediction = model.predict(hist.reshape(1, -1))

    # save the prediction for each image
    predictions.append((imagePath, prediction[0]))
    # cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
    # cv2.imshow("Image", image)
    cv2.waitKey(0)

recognised = 0
# see results
for predict in predictions:
    print(predict)
    if predict[0].split('_')[0] == predict[1].split('_')[0]:
        recognised += 1
    print(f"I classifed {predict[0]} as {predict[1]}")


percentage = recognised/50 * 100
print(percentage)
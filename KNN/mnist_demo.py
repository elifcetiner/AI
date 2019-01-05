from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn

if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.cross_validation import train_test_split

else:
	from sklearn.model_selection import train_test_split

mnist = datasets.load_digits()

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
	mnist.target, test_size=0.25, random_state=42)

(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
	test_size=0.1, random_state=84)

print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

kVals = range(1, 30, 2)
accuracies = []

for k in range(1, 30, 2):
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainData, trainLabels)

	score = model.score(valData, valLabels)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)


i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
																	   accuracies[i] * 100))

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))


for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
	image = testData[i]
	prediction = model.predict(image.reshape(1, -1))[0]

	
	image = image.reshape((8, 8)).astype("uint8")
	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)


	print("I think that digit is: {}".format(prediction))
	cv2.imshow("Image", image)
	cv2.waitKey(0)

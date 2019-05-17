from sklearn import datasets,metrics
from sklearn.svm import SVC
from scipy import misc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

digits = datasets.load_digits()
print(digits)

features = digits.data
labels = digits.target
print(features,labels)

print(digits.images.dtype)

image = cv2.imread("/home/jaydeep/Desktop/Training/ML/hand writing/09.jpg")
#print(image.shape)
print(image.dtype)

image2 = cv2.resize(image,(8,8))
#print(image2.shape)

image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#print(image3.shape)
#print(image3.dtype)


data = np.asarray(image2, dtype="float64" )
print(data)
#print(data.dtype)

image4 = misc.bytescale(data,high=16,low=0)
print(image4)

n_samples = len(digits.images)
data2 = digits.images.reshape((n_samples, -1))
print(data2)

print(data2.shape)
print(data2.dtype)
print(np.shape(data2))



classifier = RandomForestClassifier(n_estimators=100, criterion='entropy')
classifier.fit(data2,digits.target)

x_test = []

for eachRow in image4:
    for eachPixel in eachRow:
        x_test.append(sum(eachPixel)/3.0) 
   

data3=(np.array(x_test))
data4 = data3.reshape(1,64)
print(data4.shape)
predicted = classifier.predict(data4)
print(predicted)











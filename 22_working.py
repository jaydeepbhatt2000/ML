from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np


digits = datasets.load_digits()
#print(digits)

features = digits.data
labels = digits.target
#print(features,labels)

clf = SVC(gamma=0.001)
clf.fit(features,labels)

print(features.shape)
# print(clf.predict([features[-2]]))

img = misc.imread("/home/jaydeep/Desktop/Training/ML/hand writing/07.jpg")
img = misc.imresize(img,(8,8)) # resizing image to 8x8 pixel bcs daata set is in 8x8 format
#print(img.dtype)    ## to check datatype of image               
#print(img)

#print(digits.images.dtype) ## to check data type of dataset

img = img.astype(digits.images.dtype) ## convert datatype of image 
#print(img.dtype)      ## check datatype after conversion 

##print(features[-1])   ##it rnges from 0 to 16 
##print(img)              ## it ranges from 0 to 255

img = misc.bytescale(img,high=16,low=0) ##scaling image pixel values
#print(img)  ## check updated value 


x_test = []

for eachRow in img:
    for eachPixel in eachRow:
        x_test.append(sum(eachPixel)/3.0) 
        
##print(x_test) 

print(clf.predict([x_test]))



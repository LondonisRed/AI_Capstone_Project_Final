# -*- coding: utf-8 -*-
"""

AI project

Raphael Fortin          2023T015
Mael Ditsch             2023T016
Ayoub Ala Mostafa       2023T011
Nguyen Huu Trung Kien   20215216
Nguyen Thanh Tung       20226071


Handwritten digit recognition

Main
"""
from PIL import Image
import cv2
import AI_project_lib as lib
import numpy as np

#%% Importing data

taille_image = 28                                               # Defining the image size
image_pixels = taille_image * taille_image                      # Calculating the total number of pixels in the image
train_data = np.loadtxt("mnist_train.csv", delimiter=",")       # Loading training data from the "mnist_train.csv" file
test_data = np.loadtxt("mnist_test.csv", delimiter=",")         # Loading test data from the "mnist_test.csv" file

#%% Error Limitation and Data Extraction to Desired Size
trainSize, testSize = 1000, 1000                                        # Defining the size of the training set and test set

fac = 0.99 / 255                                                        # Scaling factor to normalize pixel values between 0.01 and 1
train_imgs = np.asfarray(train_data[:trainSize, 1:]) * fac + 0.01       # Converting training images to a float array and scaling the pixels
train_labels = np.asarray(train_data[:trainSize, 0])                    # Converting training labels to a numpy array
test_imgs = np.asfarray(test_data[:testSize, 1:]) * fac + 0.01          # Converting test images to a float array and scaling the pixels
test_labels = np.asarray(test_data[:testSize, 0])                       # Converting test labels to a numpy array

#%% Learning

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""  WARNING DON'T LAUNCH THIS PART
    The files are already created  """
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import time 

# List of digits from 0 to 9
digits = [i for i in range(10)] 

# Parameters for learning
rho = 10**(-13)
it = 10000

# Loop for each digit
for i in range(10):
    t1 = time.time()
    
    # Learning the digit i using the learn() function
    A, b, train_final = lib.learn(i, rho, it, train_labels, train_imgs) 
    
    # Saving matrices A, b, and train_final for digit i
    np.save("A"+str(i), A)
    np.save("b"+str(i), b)
    np.save("train_final"+str(i), train_final)
    
    t2 = time.time()
    
    # Displaying information about the current iteration
    print("iteration: {} \ntime: {} seconds".format(i, t2 - t1))

#%% Detection of 0

""" You can skip this program beacause we will launch it agian after """

# Loading matrices A0, b0, and train_final0 for digit 0
A0 = np.load("A0.npy")
b0 = np.load("b0.npy")
train_final0 = np.load("train_final0.npy")

# Test and confusion matrix for 0
Conf = np.zeros((2, 2)).astype('int')

# Loop for each test image
for i in range(testSize):
    x = test_imgs[i, :]
    lx = test_labels[i].astype('int')
    
    # If the label is 0, set the actual index to 1; otherwise, set it to 0
    if lx == 0:
        actual_index = 1
    else:
        actual_index = 0
    
    # Prediction for the image x using matrices A0, b0, and train_final0
    c = lib.prediction(x, A0, b0, train_final0, trainSize)
    
    # Updating the confusion matrix
    Conf[c, actual_index] += 1

# Display the confusion matrix and its information
print(Conf)
print("total success rate: ", np.trace(Conf) / testSize)
print("error rate: ", 1 - np.trace(Conf) / testSize)
print("Sensitivity: ", Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]))

#%% Test and confusion matrix for all digits


trainSize, testSize = 1000, 1000                                        # Defining the size of the training set and test set

fac = 0.99 / 255                                                        # Scaling factor to normalize pixel values between 0.01 and 1
train_imgs = np.asfarray(train_data[:trainSize, 1:]) * fac + 0.01       # Converting training images to a float array and scaling the pixels
train_labels = np.asarray(train_data[:trainSize, 0])                    # Converting training labels to a numpy array
test_imgs = np.asfarray(test_data[:testSize, 1:]) * fac + 0.01          # Converting test images to a float array and scaling the pixels
test_labels = np.asarray(test_data[:testSize, 0])                       # Converting test labels to a numpy array

# Lists to store matrices A, b, and train_final for each digit
listA = []
listb = []
listT = []

# Loop for each digit
for i in range(10):
    
    # Loading matrices A, b, and train_final for digit i
    listA.append(np.load("A"+str(i)+".npy"))
    listb.append(np.load("b"+str(i)+".npy"))
    listT.append(np.load("train_final"+str(i)+".npy"))

#%% Matrice de confusion pour la reconnaissance de chaque chiffre

lib.conf_tous(testSize,trainSize,test_imgs,test_labels,listA,listb,listT)

    
#%% Matrice de confusion pour la reconnaissance de tous les chiffres

lib.mat_conf(test_imgs,test_labels,trainSize,listA,listb,listT)

#%% Matrice de confusion avec la méthode sans noyaux

W=lib.fonc_W(train_imgs, train_labels)
lib.mat_conf_sans_kernel(W,test_imgs, test_labels)

#%% Demonstration of the prediction

for i in range(10):
    x = test_imgs[i]
    lx = test_labels[i].astype('int')
    result = lib.f_pred(trainSize, x, listA, listb, listT)
    print("true = {} , prediction = {}\nindex = {}\n".format(lx,result,i))
    
#%% Diplay and prediction for a given digit i

i = 8

img = test_imgs[i].reshape((28,28))
img_resized = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Visualisation agrandie', img_resized)

x = test_imgs[i]
lx = test_labels[i].astype('int')
result = lib.f_pred(trainSize, x, listA, listb, listT)
print("true : {} , prediction : {}\n".format(lx,result))

#%% Test for a document

# Charger l'image à partir du fichier
image_path = 'image.png'
image = Image.open(image_path)

# Convertir l'image en une matrice numpy
matrix = np.array(image)


test_image = (matrix[:,:,0] + matrix[:,:,1] + matrix[:,:,2]).reshape((784,))

result = lib.f_pred(trainSize, test_image, listA, listb, listT)
print(result)

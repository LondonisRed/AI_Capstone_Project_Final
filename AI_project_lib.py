# -*- coding: utf-8 -*-
"""

AI project

Raphael Fortin          2023T015
Mael Ditsch             2023T016
Ayoub Ala Mostafa       2023T011
Nguyen Huu Trung Kien   20215216
Nguyen Thanh Tung       20226071


Handwritten digit recognition

Lib
"""

import numpy as np

#%%

def kern(x, y):
    c = 1
    return (c + x.T @ y)**2

def KforC(X, U, V):
    # Here we create the Gram matrix with '-' signs on V to simplify the C matrix
    p = np.shape(U)[0]
    n = np.shape(X)[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < p:
                K[i, j] = -kern(X[j], U[i])
            else:
                K[i, j] = kern(X[j], V[i - p])
    return K

def matC(K, p, q):
    # Creation of the C matrix adapted to the problem
    sup = np.ones((p, 1))
    inf = -np.ones((q, 1))
    vecteurUn = np.vstack((sup, inf))
    C = np.hstack((K, vecteurUn))
    return C

def proj(x):
    # Creating y from x by projection
    y = x
    for j in range(0, len(x)):
        y[j] = np.maximum(0, x[j])
    return y

def Uzawa(C, rho, it):
    # Uzawa algorithm using the C matrix defined with kernels
    mu = np.ones((np.shape(C)[0], 1))
    for i in range(0, it):
        x = (-1) * (C.T @ mu)
        mu = proj(mu + rho * (C @ x + np.ones((np.shape(C)[0], 1))))
    return mu

def vecA(C, mu):
    # Extracting coefficients A from mu
    A = (-C.T @ mu)[:-1, :]
    return A

def indiceNsupport(mu):
    # This function detects all indices where mu is equal to zero
    indiceNsupport = np.where(mu == 0)[0]
    return indiceNsupport

def scalB(X, A, U, V):
    # Calculation of the scalar b
    p = np.shape(U)[0]
    q = np.shape(V)[0]
    b = 0
    for i in range(p):
        for j in range(p + q):
            b += (1/p) * A[j] * kern(U[i], X[j])
    for i in range(q):
        for j in range(p + q):
            b += (1/q) * A[j] * kern(V[i], X[j])
    return b[0]

def ensembles(nombredetection, train_labels, train_imgs):
    # This function defines matrices U and V as well as the training matrix classified with U on top and V on the bottom
    indice_u = np.where(train_labels == nombredetection)
    indice_v = np.where(train_labels != nombredetection)
    U = train_imgs[indice_u]
    V = train_imgs[indice_v]
    return np.vstack((U, V)), U, V

def learn(nombredetection, rho, it, train_labels, train_imgs):
    # The learn function is the training function that returns coefficients A, scalar b, and the associated training matrix each time
    train_final, U, V = ensembles(nombredetection, train_labels, train_imgs)
    p = np.shape(U)[0]
    q = np.shape(V)[0]
    
    K = KforC(train_final, U, V)
    C = matC(K, p, q)
    mu = Uzawa(C, rho, it)
    A = vecA(C, mu)
    b = scalB(train_final, A, U, V)
    return A, b, train_final

def prediction(Z, A, b, train_final, trainSize):
    # The prediction function detects if a digit is the desired digit or not
    s = 0
    for i in range(trainSize):
        s += A[i] * kern(Z, train_final[i])
    s = s - b
    return int(s > 0)

def conf_tous(testSize, trainSize, test_imgs, test_labels, listA, listb, listT):
    # This function displays all confusion matrices for the detection of each digit separately
    for k in range(10):
        Conf = np.zeros((2, 2)).astype('int')
        for i in range(testSize):
            x = test_imgs[i, :]
            lx = test_labels[i].astype('int')
            if lx == k:
                indiceReel = 1
            else:
                indiceReel = 0
            c = prediction(x, listA[k], listb[k], listT[k], trainSize)
            Conf[c, indiceReel] += 1
        print(Conf)
        print("Total success rate for {}:".format(k), np.trace(Conf) / testSize)
        print("Error rate for {}:".format(k), 1 - np.trace(Conf) / testSize)
        print("Sensitivity for {}:".format(k), Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]), "\n")

def f_pred(trainSize, Z, listA, listb, listT):
    # The f_pred function detects which handwritten digit the tested image is closest to based on the provided data
    s = 0
    sValues = []

    for j in range(10):
        for i in range(trainSize):
            s += listA[j][i] * kern(Z, listT[j][i])
        s = s - listb[j]
        if s[0] > 0:
            sValues.append(s)
            s = 0
        else:
            sValues.append(-1000)
            s = 0

    return sValues.index(max(sValues))

def mat_conf(test_imgs, test_labels, trainSize, listA, listb, listT):
    # This function returns the 10x10 confusion matrix to see which prediction is made for each digit
    line, col = np.shape(test_imgs)
    Conf = np.zeros((10, 10)).astype('int')
    for i in range(line):
        x = test_imgs[i, :]
        lx = test_labels[i].astype('int')
        c = f_pred(trainSize, x, listA, listb, listT)
        Conf[lx, c] += 1
    print(Conf)
    print("Total success rate:", np.trace(Conf) / line)

# Case without kernels
def fonc_W(data, labels):
    D = np.hstack((data, np.ones((np.shape(data)[0], 1))))
    B = np.zeros((np.shape(D)[0], 10))
    for i in range(np.shape(D)[0]):
        B[i, int(labels[i])] = 1
    rho = 1
    W = np.linalg.inv(D.T @ D + rho * np.eye(np.shape(D)[1])) @ D.T @ B
    return W

def f_pred_sans_kernel(img, W):
    x = np.hstack((img, 1)).reshape(-1, 1)
    q = x.T @ W
    return np.argmax(q)

def mat_conf_sans_kernel(W, test_imgs, test_labels):
    line, col = np.shape(test_imgs)
    Conf = np.zeros((10, 10)).astype('int')
    for i in range(line):
        x = test_imgs[i, :]
        lx = test_labels[i].astype('int')
        c = f_pred_sans_kernel(x, W)
        Conf[lx, c] += 1
    print(Conf)
    print("Total success rate:", np.trace(Conf) / line)

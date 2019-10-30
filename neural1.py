import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
data = sio.loadmat("ex3data1.mat")
x = data['X'] 
theta = sio.loadmat("ex3weights.mat")
weights1 = theta['Theta1']
weights2=theta['Theta2']
y = np.squeeze(data['y'])
np.place(y, y == 10, 0) 
numExamples = x.shape[0] 
numFeatures = x.shape[1] 
numLabels = 10
Y = np.reshape(y, (-1, 1))
weights1.shape
weights2.shape
X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
X[:, 1:] = x
X.shape
def sigmoid(x, weights):
  predictions = np.dot(x, weights)
  return 1/(1+np.exp(-predictions))
hypo1=np.zeros((5000,25))
hypo1=sigmoid(X,weights1.T)
hypo1.shape
Hypo1 = np.ones(shape=(hypo1.shape[0],hypo1.shape[1] + 1))
Hypo1[:, 1:] = hypo1
Hypo1.shape
hypo2=np.zeros((26,10))
hypo2=sigmoid(Hypo1,weights2.T)
for i in range(5000):
    for j in range(10):
        if hypo2[i][j]==np.amax(hypo2[i],axis=0):
            hypo2[i][j]=1
        else:
            hypo2[i][j]=0
print(hypo2)

count=0
predictlabel=np.zeros((5000,1))
for i in range(5000):
    for j in range(10):
        if hypo2[i][j]==1:
            if j==9:
                predictlabel[i]=0
            else:
                predictlabel[i]=j+1
difference=predictlabel-Y
for i in range(5000):
    if difference[i]==0:
        count+=1
print(count/5000*100)


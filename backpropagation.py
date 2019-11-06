import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
def derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
data = sio.loadmat("ex4data1.mat")
x = data['X'] 
y = np.squeeze(data['y'])
np.place(y, y == 10, 0) 
#numExamples = x.shape[0] 
#numFeatures = x.shape[1] 
#numLabels = 10
Y = np.reshape(y, (-1, 1))
def sigmoid(z):
  return 1/(1+np.exp(-z))
def backprop(x,y,weights1,weights2):
 z2=np.dot(x,weights1)
 z2=np.append(np.ones((1,1)),z2,axis=1)
 a2=sigmoid(z2)
 z3=np.dot(a2,weights2)
 a3=sigmoid(z3)
 del3=a3-y
 del2=np.dot(del3,weights2.T)*derivative(z2)
 return del3,del2,a3,a2
weights1=np.random.rand(401,25)*2*0.12-0.12
weights2=np.random.rand(26,10)*2*0.12-0.12
classes = np.unique(y)
binary_y=[]
for c in classes:
    binary_y.append(np.where(y==c,1,0))
binary_y1=np.asarray(binary_y)
ytrain=binary_y1.T
X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
X[:, 1:] = x
for m in range(2000):
 delta1=np.zeros((401,25))
 delta2=np.zeros((26,10))
 for i in range(5000):
  (del3,del2,a3,a2)=backprop(X[i].reshape(1,401),ytrain[i].reshape(1,10),weights1,weights2)
  del2=np.delete(del2,0,1)
  delta2= delta2+np.dot(a2.T,del3)
  delta1= delta1+np.dot(X[i].reshape(1,401).T,del2)
 weights2-=1*(delta2/5000)
 weights1-=1*(delta1/5000)
 z2=np.dot(X,weights1)
 z2=np.append(np.ones((5000,1)),z2,axis=1)
 a2=sigmoid(z2)
 z3=np.dot(a2,weights2)
 a3=sigmoid(z3)
  #print(weights1)
  #print(weights2)
 difference=(a3-ytrain)**2
 print(difference.mean())
for i in range(5000):
	for j in range(10):
		if a3[i][j]==np.amax(a3[i],axis=0):
			a3[i][j]=1
		else:
			a3[i][j]=0
count=0
predictlabel=np.zeros((5000,1))
for i in range(5000):
    for j in range(10):
        if a3[i][j]==1:
            predictlabel[i]=j
difference=predictlabel-Y
for i in range(5000):
    if difference[i]==0:
        count+=1
print(count/5000*100)

			


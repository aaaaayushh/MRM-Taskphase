import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
data = sio.loadmat("ex3data1.mat")
x = data['X'] 
y = np.squeeze(data['y'])
np.place(y, y == 10, 0) 
numExamples = x.shape[0] 
numFeatures = x.shape[1] 
numLabels = 10
Y = np.reshape(y, (-1, 1))
def predict(x, weights):
  predictions = np.dot(x, weights)
  return 1/(1+np.exp(-predictions))
def cost_function(x, y, weights, lamb):
    obser= len(y)
    predictions= predict(x,weights)
    class1_cost= -y*np.log(predictions)
    class2_cost= (1-y)*np.log(1-predictions)
    reg=weights**2
    reg=reg.sum()
    cost= class1_cost-class2_cost+((lamb*reg)/obser)
    cost=cost.sum()/obser
    return cost
X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
X[:, 1:] = x

weights = np.random.rand(401,10)
#weight.shape
#weights= np.reshape(weight, (-1, 1))
weights.shape

def update_weights(x,y,weights,lr,lamb=100):
    n=len(x)
    predictions=predict(x,weights)
    gradient=np.dot(x.T, (predictions-y))
    gradient/=n
    gradient*=lr
    weights-=gradient+(lamb*weights/5000)
    return weights


classes = np.unique(y)
print(classes)
classes.shape

binary_y=[]
for c in classes:
    binary_y.append(np.where(y==c,1,0))
print(binary_y)
binary_y1=np.asarray(binary_y)
binary_y1.shape

def train(x, y,weights, max_iter=1000, alpha=0.1):
    classes = np.unique(y)
    costs = []
    for c in classes:
        for j in range(max_iter):
            weights=(update_weights(x,y,weights,lr=0.1))
            costs.append(cost_function(x,y,weights,lamb=100))
            print(weights)
    return weights, classes, costs

plt.plot(cost)

predictions=predict(X,weights)
for i in range(5000):
    for j in range(10):
        if predictions[i][j]==np.amax(predictions[i],axis=0):
            predictions[i][j]=1
        else:
            predictions[i][j]=0
print(predictions)

count=0
predictlabel=np.zeros((5000,1))
for i in range(5000):
    for j in range(10):
        if predictions[i][j]==1:
            predictlabel[i]=j
difference=predictlabel-Y
for i in range(5000):
    if difference[i]==0:
        count+=1
print(count/5000*100)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
column_names=['MARK1','MARK2','ADMISSION']
df=pd.read_csv('ex2data2.txt', sep=",",names=column_names)
print(df)
df.shape
x=pd.DataFrame(df.iloc[:,:-1])
y=pd.DataFrame(df.iloc[:,-1])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
y_pred=logmodel.predict(x_test)
print(logmodel.score(x_test,y_test))
print(logmodel.score(x_train,y_train))



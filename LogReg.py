import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



class Perceptron :
    
    #Parameters intialization
    def __init__(self,eta=0.01,n_iter=10,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
        
    #fiting the parameters 
    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state)
        self.w=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1]) #Random Weight generation 
        self.errors=[]
        for _ in range (self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w[1:]+=update*xi
                self.w[0]+=update
                errors+=int(update!=0.0)
            self.errors.append(errors)
        return self
    
    
    def net_input(self,X):
        return np.dot(X,self.w[1:])+self.w[0]
    
    #prediction 
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,-1)

s=os.path.join('https://archive.ics.uci.edu/','ml/', 'machine-learning-databases/', 'iris/', 'iris.data')
data=pd.read_csv(s,header=None,encoding='utf-8')

#target
y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)

#features
X=data.iloc[0:100,[0,2]].values

#visualization
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
perceptron=Perceptron(eta=0.1,n_iter=10)
perceptron.fit(X,y)
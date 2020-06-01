import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class AdalineGD(object):
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    
    def fit(self, X, y):
        rgen=np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost=[]
        
        for i in range(self.n_iter):
            net_input=self.net_input(X)
            output=self.activation(net_input)
            errors=y-output
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost.append(cost)
        return self
    
    def net_input(self, X):
        #calculates the net input 74414
        return np.dot(X, self.w[1:])+self.w[0]
    
    def activation(self, X):
        #computes the linear activation 
        return X
    
    def predict(self, X):
        #return class label after unit step 
        return np.where(self.activation(self.net_input(X)) >=0.0, 1, -1)
    

s=os.path.join('https://archive.ics.uci.edu/','ml/', 'machine-learning-databases/', 'iris/', 'iris.data')
data=pd.read_csv(s,header=None,encoding='utf-8')

y=data.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=data.iloc[0:100,[0,2]].values

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(10,5))

adaline1=AdalineGD(n_iter=10, eta= 0.01).fit(X,y)
ax[0].plot(range(1,len(adaline1.cost)+1), (adaline1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Sum-Squared Error')
ax[0].set_title('learning rate =0.01')

adaline2=AdalineGD(n_iter=10, eta= 0.0005).fit(X,y)
ax[1].plot(range(1,len(adaline1.cost)+1), np.log10(adaline2.cost), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-Squared Error')
ax[1].set_title('learning rate =0.0005')


    
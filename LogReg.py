import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

iris=datasets.load_iris() #gives an array of sepal length ,sepal width, petal length ,petal width
X = iris.data[:, :4]
y = iris.target       # setosa =0(50 samples), virginica=1(50 samples) ,versicolor=2(50 samples) [total_length=150]
Y=iris.target_names
##print(Y)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 #the + and - 0.5 is given for the purpose of graph
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 #the + and - 0.5 is given for the purpose of graph

#Data Visualization
plt.figure()
plt.subplots(1,1,figsize=(10,8))
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.scatter(X[:49, 0], X[:49, 1],color="r",label="setosa")
plt.scatter(X[49:99, 0], X[49:99, 1],color="b",label="virginica")
plt.scatter(X[100:149, 0], X[100:149, 1],color="g",label="versicolor")
plt.legend()
plt.subplots(1,1,figsize=(10,8))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.scatter(X[:49, 2], X[:49, 3],color="r",label="setosa")
plt.scatter(X[49:99, 2], X[49:99, 3],color="b",label="virginica")
plt.scatter(X[100:149, 2], X[100:149, 3],color="g",label="versicolor")
plt.legend()
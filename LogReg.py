import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris=datasets.load_iris() #gives an array of sepal length ,sepal width, petal length ,petal width
X = iris.data[:, :4]
y = iris.target       # setosa =0(50 samples), virginica=1(50 samples) ,versicolor=2(50 samples) 
Y=iris.target_names

print(pd.DataFrame(X,columns=["sepal_length","sepal_width","petal_length","petal_width",]))

log_reg=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
log_reg.fit(X_train,y_train)
log_reg.predict(X_test)
prediction=log_reg.predict([[1,3,3,4]])
print("The accuracy of the iris data is :",int(log_reg.score(X_test,y_test)*100),"%")
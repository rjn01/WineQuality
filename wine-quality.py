
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv("wine.csv")
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
y=df.iloc[:,-1].values
x=df.iloc[:,:-1]
y=df['quality']

#ENCODING 
from sklearn.preprocessing import LabelEncoder              
lx=LabelEncoder()
y = lx.fit_transform(y)

#SPLITTING DATA
from sklearn.model_selection import train_test_split   
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#SCALING
from sklearn.preprocessing import StandardScaler             
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
linReg=LinearRegression()
linReg.fit(x_train,y_train)

y_pred=linReg.predict(x_test)
from sklearn.metrics import explained_variance_score         #(ACCURACY)
a=explained_variance_score(y_test,y_pred)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logReg=LogisticRegression()
logReg.fit(x_train,y_train)

y_pred1=logReg.predict(x_test)

from sklearn.metrics import confusion_matrix                  #(ACCURACY)
cm=confusion_matrix(y_test,y_pred1)

# GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred2=gnb.predict(x_test)

from sklearn.metrics import accuracy_score                    #(ACCURACY)
ac=accuracy_score(y_test,y_pred2)

#SUPPORT VECTOR MACHINE 
from sklearn.svm import SVC
svc1=SVC(C=5,kernel='rbf',gamma=0.8)  
svc1.fit(x_train,y_train)

y_pred3=svc1.predict(x_test)

from sklearn.metrics import accuracy_score                     #(ACCURACY)
acc4=accuracy_score(y_test,y_pred3)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred4=dt.predict(x_test)

from sklearn.metrics import accuracy_score #(ACCURACY)
acc3=accuracy_score(y_test,y_pred4)

#K-NEAREST NEIGHBOUR
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=10,algorithm='ball_tree')  #USING BALL TREE ALGO
clf.fit(x_train,y_train)
clf1=KNeighborsClassifier(n_neighbors=10,algorithm='brute')     #USING BRUTE FORCE ALGO
clf1.fit(x_train,y_train)
clf2=KNeighborsClassifier(n_neighbors=10,algorithm='kd_tree')   #USING KD TREE
clf2.fit(x_train,y_train)

#Prediction
y_pred5=clf.predict(x_test)
y_pred6=clf1.predict(x_test)
y_pred7=clf2.predict(x_test)

from sklearn.metrics import accuracy_score                       #(ACCURACY)
accc=accuracy_score(y_test,y_pred5)
acc1=accuracy_score(y_test,y_pred6)
acc2=accuracy_score(y_test,y_pred7)


# Printing values of accuracy of diffrent algorithms

print('LINEAR REGRESSION',a)
print('LOGISTIC REGRESSION',cm)
print('NAIVE BAYES',ac)
print('SVC',acc4)
print('DT',acc3)
print('BALL',accc,'BRUTE',acc1,'KDTREE',acc2,end='',sep='\n')

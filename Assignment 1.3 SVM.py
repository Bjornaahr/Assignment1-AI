# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns
import time


#%%
def digitsRecognition(X_train, X_test, y_train, y_test):

    C = 100

    clf = svm.SVC(gamma=0.001, C=C) # Gamma is step size
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print("Score SVM: ", score)

    plt.show()
    return time.time()

#%%
def randomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=500, max_depth=1000)
    rf.fit(X_train, y_train)

    score = rf.score(X_test, y_test)

    print("Score randomForest: ", score)
    return time.time()

#%%
def logisticRegression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial', max_iter=1000)
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print ("Score Logistic Regression: ", score)
    return time.time()



#%%
X, y = datasets.load_iris(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
start_time = time.time() # Start time of prediction

end_time = digitsRecognition(X_train, X_test, y_train, y_test)

print("Time to exectute SVM on iris: %s seconds" % (end_time - start_time))

#%%
start_time = time.time() # Start time of prediction

end_time = randomForest(X_train, X_test, y_train, y_test)

print("Time to exectute randomForest on iris: %s seconds" % (end_time - start_time))

#%%
start_time = time.time() # Start time of prediction

end_time = logisticRegression(X_train, X_test, y_train, y_test)

print("Time to exectute logisticRegression on iris: %s seconds" % (end_time - start_time))



#%%

train_samples = 60000

X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
start_time = time.time() # Start time of prediction

end_time = digitsRecognition(X_train, X_test, y_train, y_test)

print("Time to exectute SVM on mnist: %s seconds" % (end_time - start_time))

#%%
start_time = time.time() # Start time of prediction

end_time = randomForest(X_train, X_test, y_train, y_test)

print("Time to exectute randomForest on mnist: %s seconds" % (end_time - start_time))

#%%
start_time = time.time() # Start time of prediction

end_time = logisticRegression(X_train, X_test, y_train, y_test)

print("Time to exectute logisticRegression on mnist: %s seconds" % (end_time - start_time))


#%%

train_samples = 50000

X,y = fetch_openml('CIFAR_10', version=1, return_X_y=True)
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
start_time = time.time() # Start time of prediction

end_time = digitsRecognition(X_train, X_test, y_train, y_test)

print("Time to exectute SVM on cifar: %s seconds" % (end_time - start_time))


#%%
start_time = time.time() # Start time of prediction

end_time = randomForest(X_train, X_test, y_train, y_test)

print("Time to exectute randomForest on cifar: %s seconds" % (end_time - start_time))

#%%
start_time = time.time() # Start time of prediction

end_time = logisticRegression(X_train, X_test, y_train, y_test)

print("Time to exectute logisticRegression on cifar: %s seconds" % (end_time - start_time))

#%%



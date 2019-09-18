
# coding: utf-8

# In[1]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import loadtxt
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
import numpy as np


# In[2]:


data = loadtxt('pima-indians-diabetes.csv', delimiter=",")

X = data[:,0:8]
y = data[:,8]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42, test_size=0.2)


# In[3]:


model = XGBClassifier()
model.fit(Xtrain, ytrain)

y_pred = model.predict(Xtest)

accuracy = accuracy_score(ytest, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[4]:


model.fit(X,y)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()


# In[5]:


all_acc=cross_val_score(model, X, y, cv=5)
m_acc = np.mean(all_acc)
std_acc = np.std(all_acc)
print("Accuracy: mean = %.2f%% (std = %.2f%%)" % (m_acc * 100.0,std_acc * 100.0))


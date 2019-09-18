
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('HousePrice.csv')
data.head(5)


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics


# In[4]:


X=data[data.columns[:13]]
y=data[data.columns[13]]
#X.head(5)
#y.head(5)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=42,test_size = 0.3)


# In[5]:


def PolynomialRegression(degree=2,**kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))

poly = PolynomialRegression()
poly.fit(Xtrain,ytrain)

ytrain_pred = poly.predict(Xtrain)
ytest_pred = poly.predict(Xtest)


# In[6]:


print('Training set R2 is %1.2f'%metrics.r2_score(ytrain,ytrain_pred))
print('Testing set R2 is %1.2f'%metrics.r2_score(ytest,ytest_pred))


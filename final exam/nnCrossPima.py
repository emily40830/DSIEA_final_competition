
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras import models
from keras import layers


# In[13]:


data = np.loadtxt('pima-indians-diabetes.csv', delimiter=",")

X = data[:,0:8]
y = data[:,8]


# In[10]:


model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_dim=8))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# In[7]:


model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# In[19]:


# define 5-fold cross validation test
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_acc = []
for train_index, test_index in kfold.split(X,y):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    
    model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
    result = model.fit(Xtrain, ytrain, epochs=150, batch_size=10)
    scores = model.evaluate(Xtest, ytest)
    val_acc.append(scores[1])


# In[26]:


for x in range(len(val_acc)): 
    print('acc:%.2f%%' % (val_acc[x]*100))
          
import statistics
mean=statistics.mean(val_acc)
std=statistics.stdev(val_acc)
print('%.2f%%(+/-)%.2f%%'%(mean*100,std*100))


# In[25]:





# In[24]:





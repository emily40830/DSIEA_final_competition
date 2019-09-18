
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers


# In[4]:


data = np.loadtxt('pima-indians-diabetes.csv', delimiter=",")

X = data[:,0:8]
y = data[:,8]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42, test_size=0.2)


# In[8]:


model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_dim=8))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# In[11]:


model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
result = model.fit(Xtrain, ytrain, epochs=150, batch_size=10,validation_data=(Xtest,ytest))


# In[24]:


import matplotlib.pyplot as plt

acc = result.history['acc']
val_acc = result.history['val_acc']
loss = result.history['loss']
val_loss = result.history['val_loss']

epoch = range(150)


plt.plot(epoch, acc, 'bo', label='Training acc')
plt.plot(epoch, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()


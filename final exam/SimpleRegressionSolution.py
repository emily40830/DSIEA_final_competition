
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([1,2,3,4])
y = np.array([9,13,14,18])


# In[2]:


#home-made linear regression model

def estimate_coef(x, y): 
    n = np.size(x)
    avg_x, avg_y = np.mean(x), np.mean(y) 
    
    b_1 = (np.sum(y*x) - n*avg_y*avg_x)/(np.sum(x*x) - n*avg_x*avg_x)
    b_0 = avg_y - b_1*avg_x 
    
    return(b_0, b_1)

def lm_hand_made(x,y):
    
    b = estimate_coef(x,y)
    y_pred = b[0]+b[1]*x
    
    return(y_pred)

b = estimate_coef(X, y)

#From Scikit-Learn linear regression model
lm = LinearRegression()

#X.reshape(-1, 1)
#y.reshape(-1, 1)
lm.fit(X.reshape(-1, 1),y.reshape(-1, 1))


# In[3]:


print('From home-made linear regression model\nbeta0: %1.1f\nbeta1: %1.1f'%(b[0],b[1]))
print('From Scikit-Learn linear regression model\n','%1.1f\n[%1.1f]'%(lm.intercept_,lm.coef_))


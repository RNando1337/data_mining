#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd


# In[70]:


ds = pd.read_csv("DataWisata.csv")

x = ds.iloc[:, [2,4]].values
y = ds.iloc[:, [0,3]].values


# In[71]:


print(np.array(x))


# In[72]:


print(y)


# In[74]:


from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder as OHE
ct = CT(transformers=[('encoder', OHE(sparse=False), [0])], remainder='passthrough') #penamanaan "tes" bebas
x = np.array(ct.fit_transform(x))                            


# In[75]:


print(x)


# In[36]:


from sklearn.preprocessing import LabelEncoder as LE
lbl_encode = LE()
y[:, 0] = lbl_encode.fit_transform(y[:, 0])


# In[37]:


print(y)


# In[54]:


from sklearn.model_selection import train_test_split as tts
x_train, x_tes, y_train, y_tes = tts(x,y, test_size = 0.2, random_state = 1)


# In[53]:


print(x_train)


# In[40]:


print(x_tes)


# In[41]:


print(y_train)


# In[42]:


print(y_tes)


# In[55]:


from sklearn.preprocessing import StandardScaler as SS
sc = SS()
x_train[:, -1:] = sc.fit_transform(x_train[:, -1:])
x_tes[:, -1:] = sc.transform(x_tes[:, -1:])


# In[56]:


print(x_train)


# In[57]:


print(x_tes)


# In[ ]:





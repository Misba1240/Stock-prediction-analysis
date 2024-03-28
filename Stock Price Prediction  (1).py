#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Name: Misba Sharif Inamdar
#Company: Corizo
#Domain: Data Science
#Mentor-led


# In[3]:


import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[2]:


pip install tensorflow


# In[7]:


data = pd.read_excel("C:/Users/91992/Downloads/Minor Project-20240312T063352Z-001/Minor Project/Minor Project Data set (Stock Price Prediction).xlsx")
data.head()


# In[8]:


data.shape


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


data.describe()


# In[12]:


plt.figure(figsize=(15, 8))
plt.plot(data['Date'], data['Adj Close'])
plt.xticks(rotation=45)
plt.show()


# In[13]:


# Close Prices
close_prices = data['Adj Close'].values.reshape(-1, 1)


# In[14]:


train_ratio = 0.8
train_len = int(close_prices.shape[0]*train_ratio)


# In[15]:


# Scaling the data

scalar = StandardScaler()
scalar.fit(close_prices[:train_len])
scaled_data = scalar.transform(close_prices).flatten()


# In[16]:


# X, y
T = 10
D = 1
X = []
Y = []
for t in range(len(scaled_data) - T):
    x = scaled_data[t:t+T]
    X.append(x)
    y = scaled_data[t+T]
    Y.append(y)
X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = int(len(X)*train_ratio)
print(f'X shape = {X.shape}\nY shape = {Y.shape}')


# In[17]:


i = Input(shape=(T, 1))
x = LSTM(100)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
            loss = 'mse',
            optimizer=Adam(learning_rate=0.01))
r = model.fit(X[:N], Y[:N],
             epochs=80,
             validation_data=(X[N:], Y[N:]))


# In[18]:


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# In[19]:


outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.figure(figsize=(15,8))
plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()


# In[ ]:





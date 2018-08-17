
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# In[8]:


nba = pd.read_csv('https://www.dropbox.com/s/b3nv38jjo5dxcl6/nba_2013.csv?dl=0')
nba.head(10)


# In[9]:


nba.columns


# In[12]:


nba.info()


# In[17]:


nba.describe()


# In[18]:


nba.fillna(0,inplace=True)
nba.describe()


# In[22]:


nba.shape


# In[24]:


numeric_data = nba.select_dtypes(exclude='object')
numeric_data.drop(columns=['season_end'],inplace=True)


# In[25]:


numeric_data.head(10)


# In[30]:


numeric_data_normalized = (numeric_data - numeric_data.mean())/numeric_data.std()


# In[31]:


numeric_data_normalized.head(10)


# In[33]:


y = numeric_data_normalized['pts']


# In[34]:


numeric_data_normalized.drop(columns=['pts'],inplace=True)
X = numeric_data_normalized


# In[35]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[81]:


optimal_predictions = []
error = 1
converted_y = y_test * numeric_data['pts'].std() + numeric_data['pts'].mean()
for neighbours in range(1,25):
    neigh = KNeighborsRegressor(n_neighbors=neighbours,algorithm='auto')
    neigh.fit(X_train, y_train)
    predictions = neigh.predict(X_test)
    mse = (((predictions - y_test) ** 2).sum()) / len(predictions)
    print('neigbours = %d --------------' %(neighbours))
    print(mse)
    converted_optimals = predictions * numeric_data['pts'].std() + numeric_data['pts'].mean()
    print((((converted_y - converted_optimals) ** 2).sum()) / len(converted_y))
    if mse < error:
        error = mse
        optimal_predictions = predictions


# neighbours = 3 , algo = 'auto' is giving the minimum mse around 5346

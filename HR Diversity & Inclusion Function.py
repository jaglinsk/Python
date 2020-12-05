#!/usr/bin/env python
# coding: utf-8

# Build a function that identifies variables that correlate w/ protected classes to mitigate risk of unintentional biases in our HR models

# In[29]:


import pandas as pd
import math
import os
import sys
from pathlib import Path
import csv
import seaborn as sn
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


os.chdir('C:/Users/Bridget Marie Yoga/Documents/GitPy/Python')


# In[5]:


hr = pd.read_csv('HRDataset_v14.csv')


# Let's explore the data!

# In[6]:


hr.describe()


# In[51]:


hr.dtypes


# In[10]:


hr.head()


# In[88]:


#Encode for correlation
for i in hr.columns:
    if hr[i].dtype == object:
        hr[i]=hr[i].astype('category').cat.codes


# In[91]:


mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
tri_df = corrMatrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.3)]


# Look at correlations w/ sensitive variables

# In[97]:


tri_df['GenderID'].abs().sort_values(ascending=False).head()


# In[118]:


tri_df['RaceDesc'].abs().sort_values(ascending=False).head()


# In[100]:


tri_df['HispanicLatino'].abs().sort_values(ascending=False).head()


# Make it a function

# In[120]:


def corr_var(df, var, threshold):
    for i in df.columns:
        if df[i].dtype == object:
            df[i]=df[i].astype('category').cat.codes
    
    corrMatrix = df.corr()
    mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
    tri_df = corrMatrix.mask(mask)
    tri_df[var] = tri_df[var].abs().sort_values(ascending=False)
    
    # List column names of highly correlated features (r > threshold)    
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
    return to_drop
    
    
    


# In[121]:


corr_var(hr,'GenderID',0.7)


# In[ ]:





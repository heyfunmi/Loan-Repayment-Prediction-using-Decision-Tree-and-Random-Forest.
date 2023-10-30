#!/usr/bin/env python
# coding: utf-8

# ### Hey there, this a classification machine learning project using Decision Tree and Random Forest to predict if an individual will repay a loan or not.

# In[1]:


#Importing the neccessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing the dataset
loan_dataset = pd.read_csv("loan-data.csv")


# In[3]:


#A peep into what the dataset looks like
loan_dataset.head()


# In[4]:


#Checking the number of rows and columns present in the dataset
loan_dataset.shape


# In[5]:


#Statistical describtion of the dataset
loan_dataset.describe()


# In[6]:


loan_dataset.info()


# ### Data Cleaning

# In[7]:


loan_dataset.isnull().sum()


# ### Encoding the categorical features.
# The "purpose" column is the only cetagorical feature here, so it needs to be transformed.
# 

# In[8]:


cat_feats=['purpose']


# In[9]:


loan =pd.get_dummies(loan_dataset,columns=cat_feats,drop_first=True)


# ### Train Test Split

# In[10]:


from sklearn.model_selection import train_test_split 


# In[11]:


X = loan.drop('not.fully.paid',axis=1)
y = loan['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ### Training the Decision Tree Model

# In[12]:


from sklearn.tree import DecisionTreeClassifier


# In[13]:


tree =DecisionTreeClassifier()
tree.fit(X_train,y_train)


# ### Checking the accuracy of the Decision Tree Model

# In[14]:


from sklearn.metrics import accuracy_score
y_pred = tree.predict(X_test)


# In[15]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score: {:.2f}%".format(accuracy * 100))


# ### Training the Random Forest model

# In[16]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# ### Checking the accuracy of the Decision Tree Model¶

# In[17]:


y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score: {:.2f}%".format(accuracy * 100))


# ## Thats it for this project!

# In[ ]:





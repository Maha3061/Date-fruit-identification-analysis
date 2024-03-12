#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# ## Preparing the data

# In[2]:


pip install openpyxl


# In[3]:


df = pd.read_excel("E:\VIT\G1_DMT\Date_Fruit_Datasets.xlsx")
df


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[6]:


df['Class'].unique()


# In[7]:


df['Class'].value_counts()


# In[8]:


sns.displot(df['Class'])


# In[9]:


df['Class'].replace(['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY'],
                        [0, 1, 2, 3, 4, 5, 6], inplace=True)


# In[10]:


df


# ##  Splitting the data

# In[11]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=100)


# ## Decision Tree Classifier
# 

# In[12]:


DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=42)
DecisionTreeClassifierModel.fit(X_train, y_train)

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))


# In[13]:


y_pred_DT = DecisionTreeClassifierModel.predict(X_test)
CM_DT = confusion_matrix(y_test, y_pred_DT)
sns.heatmap(CM_DT, center=True)
plt.show()
print('Confusion Matrix is\n', CM_DT)


# ## RandomForestClassifier Model

# In[14]:


RandomForestClassifierModel = RandomForestClassifier(criterion = 'entropy', max_depth=5, n_estimators=500, random_state=42)
RandomForestClassifierModel.fit(X_train, y_train)

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))


# In[15]:


y_pred_RF = RandomForestClassifierModel.predict(X_test)
CM_RF = confusion_matrix(y_test, y_pred_RF)

sns.heatmap(CM_RF, center=True)
plt.show()

print('Confusion Matrix is\n', CM_RF)


# ## GradientBoostingClassifier Model

# In[16]:


GBCModel = GradientBoostingClassifier(n_estimators=500, max_depth=2, learning_rate=0.01, random_state=42)
GBCModel.fit(X_train, y_train)
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))


# In[17]:


y_pred_GB = GBCModel.predict(X_test)
CM_GB = confusion_matrix(y_test, y_pred_GB)

sns.heatmap(CM_GB, center=True)
plt.show()

print('Confusion Matrix is\n', CM_GB)


# ## VotingClassifier Model

# In[18]:


VotingClassifierModel = VotingClassifier(estimators=[('GBCModel',GBCModel),
                                                     ('RFCModel',RandomForestClassifierModel),
                                                     ('TDCModel',DecisionTreeClassifierModel)],
                                         voting='soft')
VotingClassifierModel.fit(X_train, y_train)
print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))


# In[19]:


y_pred_V = VotingClassifierModel.predict(X_test)
CM_V = confusion_matrix(y_test, y_pred_V)
sns.heatmap(CM_V, center=True)
plt.show()
print('Confusion Matrix is\n', CM_V)


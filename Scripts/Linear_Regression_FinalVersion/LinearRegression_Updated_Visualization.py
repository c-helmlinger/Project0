#!/usr/bin/env python
# coding: utf-8

# 

# # Installation

# In[4]:


#!pip install matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# for model building

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import RFE

# for model evaluation

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # Introduction of Data

# In[80]:


df_origin = pd.read_csv("/Users/jessicazhang/Desktop/dataCleaned.csv")


# In[78]:


df_origin.columns


# In[75]:


df_origin.info()


# In[76]:


df_origin.head()


# In[77]:


df_origin.shape


# In[78]:


df_origin.describe()


# In[79]:


df_origin.corr()


# # Analyzing Data

# ### Convert the YearBuilt

# In[80]:


df_origin["YearBuilt"] = 2021 - df_origin["YearBuilt"]
df_origin.head()


# ### Outlier detection {Method 1}

# #### Landsize

# In[81]:


print(df_origin["Landsize"].describe())


# #### Building Area

# In[83]:


print(df_origin["BuildingArea"].describe())


# #### Distance

# In[53]:


print(df_origin["Distance"].describe())


# #### YearBuilt

# In[82]:


print(df_origin["YearBuilt"].describe())


# ### Method 2: Histogram

# In[84]:


df_origin.hist()


# ### Method 3: Other diagrams

# In[86]:


# Boxplot: Example
sns.boxplot(x='YearBuilt',y='Price',data=df_origin)

# comment: from the graph, it shows little correlation


# In[89]:


# Scatterplot
sns.lmplot(x='Landsize', y='Price', data=df_origin)
sns.lmplot(x='BuildingArea', y='Price', data=df_origin)
sns.lmplot(x='YearBuilt', y='Price', data=df_origin)
sns.lmplot(x='Distance', y='Price', data=df_origin)


# ### Method 4: Formula (Data Size)

# In[180]:


# Checking the percentage of outlier for all the numerical columns
# Formula: x > Q3 + 1.5IQR or x < Q1 -1.5IQR

# Single
# Calculate or Find on the graph
Q1 = df_origin.BuildingArea.quantile(0.25)
Q3 = df_origin.BuildingArea.quantile(0.75)
print("Q1:",Q1)
print("Q3:", Q3)

# IQR (Inter Quatile Range) = Q3 - Q1
IQR = Q3 - Q1

# Limit
Lower_limit = Q1 - 1.5*IQR
Upper_limit = Q3 + 1.5*IQR
print("Lower_limit:", Lower_limit)
print("Upper_limit:", Upper_limit)


# ### Delete outliers

# In[177]:


# (1) Delete outliers using the code? How?

deleted_indices = []
for col in df_origin.columns[6:9]:
    Q1 = df_origin[col].quantile(0.25)
    Q3 = df_origin[col].quantile(0.75)
    print("{}, Q1: {:.2f}, Q3: {:.2f}".format(col, Q1, Q3))
  
    # IQR (Inter Quatile Range) = Q3 - Q1
    IQR = Q3 - Q1

    # Limit
    Lower_limit = Q1 - 1.5*IQR
    Upper_limit = Q3 + 1.5*IQR
    
    Upper_num = (df_origin[df_origin[col] > Upper_limit]).shape[0]
    Lower_num = (df_origin[df_origin[col] < Lower_limit]).shape[0]
    outlier_rate = (Lower_num + Upper_num) / df_origin.shape[0]
    Upper_index = list((df_origin[df_origin[col] > Upper_limit]).index)
    Lower_index = list((df_origin[df_origin[col] < Lower_limit]).index)
    deleted_indices += (Upper_index + Lower_index)
    
    print("Lower_limit:{:.2f}, Upper_limit:{:.2f}, outlier_rate: {:.4f}% \n---\n".format(Lower_limit, Upper_limit, outlier_rate*100))

new_df = df_origin.drop(list(set(deleted_indices)), axis=0)

# (2) After finding the most obvious point on the graph, then you can just delete it on Excel. {Based on this one}


# # Read New .csv

# In[229]:


df = pd.read_csv("/Users/jessicazhang/Desktop/data.csv")


# In[6]:


df.shape


# # Data Visualisation

# ### Correlation in numeric data

# ##### Method 1: Heatmap

# In[132]:


# General
sns.heatmap(df.corr(), square=True, annot=True, cmap="YlGnBu")


# In[133]:


# Pick the top four
# correlation matrix
corrmat = df.corr()
k = 4 # number of variables for heatmap
cols = corrmat.nlargest(k, 'Price')['Price'].index # choose the top four features that have relationship with price
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.25)
# cmap means the color, annot short for "annotation"
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols, xticklabels=cols.values, cmap="YlGnBu")
plt.show()


# ##### Method 2: pairplot

# In[134]:


# Show correlation
# scatterplot
sns.pairplot(df)


# ##### Method 3: Enlarged scatterplots (x&y)

# In[135]:


# All the columns
for each in df.columns:
    plt.scatter(df[each], df['Price'])
    plt.xlabel(each)
    plt.ylabel("Melbourne Housing Price")
    plt.grid()
    plt.show()


# ### Visualizing the distribution of the numeric features

# ##### Method 1

# In[137]:


sns.distplot(df['Price'])


# In[138]:


# calculate the skewness
print("Skewness: %f" %df['Price'].skew())


# ###### Comment: Since Price is slightly right skewed, checking the distribution of transformed Price

# ##### Method 2: Histogram

# In[85]:


df.hist(bins=15)


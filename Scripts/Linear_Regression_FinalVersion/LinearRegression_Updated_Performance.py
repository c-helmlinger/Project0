#!/usr/bin/env python
# coding: utf-8

# 

# # Installation

# In[1]:


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


# In[36]:


df = pd.read_csv("/Users/jessicazhang/Desktop/data.csv")


# In[37]:


df = df.sample(frac=1, random_state=2021)


# In[38]:


df.shape


# # Adding Columns

# ## 1. for x5^0.5

# In[150]:


# x5 is the number of bathroom
# Bathroom_sqrt is the result of square of root

df['Bathroom_sqrt'] = np.sqrt(df['Bathroom'])
df.head()


# ## 2. for x5^2

# In[32]:


# Bathroom_sqr means Bathroom*Bathroom
# pow() means the square
# i.e. pow(4,2) = 4^2 = 16

df['Bathroom_sqr'] = pow(df['Bathroom'],2)
df.head()


# ## 3. for x8^0.5

# In[22]:


# x8 is the building area
# BuildingArea_sqr is the square root.

df['BuildingArea_sqrt'] = np.sqrt(df['BuildingArea'])
df.head()


# ## 4. for x8^2

# In[25]:


# BuildingArea_sqr is the square result.

df['BuildingArea_sqr'] = pow(df['BuildingArea'],2)
df.head()


# ## 5. for x9^0.5

# In[66]:


# x9 is the distance
# Distance_sqrt is the square root.

df['Distance_sqrt'] = np.sqrt(df['Distance'])
df.head()


# ## 6. for x9^2

# In[45]:


# Distance_sqr is the square.

df['Disctance_sqr'] = pow(df['Distance'],2)
df.head()


# ## 7. combination

# In[39]:


# X8 is the building area
# pow() means the square of the number
# i.e. pow(4,2) = 4*4 = 16

df['BuildingArea_sqr'] = pow(df['BuildingArea'],2)
df['Distance_sqr'] = pow(df['Distance'], 2)
df['Bathroom_sqrt']= np.sqrt(df['Bathroom'])

df.head()


# # Drop Column

# In[23]:


df = df.drop(['Rooms'], axis=1)


# In[40]:


df = df.drop(['Landsize'], axis=1)
df.head()


# # Spliting into Train and Test Data

# In[41]:


df["YearBuilt"] = 2021 - df["YearBuilt"]
df.head()


# In[42]:


# Drop the last column
X = df.drop(['Price'], axis = 1)


# In[43]:


X


# In[44]:


y = df["Price"]
print(y)


# ## Normalization

# In[9]:


from sklearn import preprocessing


# In[45]:


min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)  


# ## Model

# In[173]:


# Divide the data into two groups: training set and testing set

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=320, test_size=0.27)


# ### K Fold

# In[11]:


from sklearn.model_selection import KFold


# In[46]:


kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[47]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Linear Regression

# In[48]:


model = LinearRegression()


# In[49]:


model.fit(X_train, y_train)


# ## Model Evaluation

# ### Score

# In[16]:


from sklearn.model_selection import cross_val_score


# In[17]:


# (1) Orginal
orginal_score = cross_val_score(model, X, y, cv=5)
print("Original score:",orginal_score)


# In[33]:


# (2) Drop a column ‘Rooms’ and Add a new column: x8^0.5
drop_score = cross_val_score(model, X, y, cv=5)
print("After dropping one column:",drop_score)


# In[50]:


# (3) Drop a column & Feature Engineer (use the combination) 
combined_score = cross_val_score(model, X, y, cv=5)
print("Combination score:",combined_score)


# ## Based on the Third Model

# ### Coefficients & RMSE

# In[51]:


# y_pred_class = model.predict(X_test)
model.predict(X_test)


# In[52]:


y_test


# In[53]:


# Root Mean Squared Error

def customer_scorer(model, X, y):
    predict_y = model.predict(X)
    rmse = np.sqrt(np.power(predict_y - y, 2).sum() / len(y))
    # rmse = mean_squared_error(predict_y, y, squared=False)
    return rmse


# In[19]:


original_rmse = cross_val_score(model, X, y, cv=5, scoring=customer_scorer)
print("Original RMSE:" + (', {:.4f}' * 5).format(*original_rmse))


# In[35]:


drop_rmse = cross_val_score(model, X, y, cv=5, scoring=customer_scorer)
print("Drop RMSE:" + (', {:.4f}' * 5).format(*drop_rmse))


# In[54]:


combined_rmse = cross_val_score(model, X, y, cv=5, scoring=customer_scorer)
print("Combination RMSE:" + (', {:.4f}' * 5).format(*combined_rmse))


# ## Model Comparison

# #### Score

# In[56]:


original_score = [0.52330568, 0.5656773,  0.52000582, 0.54640202, 0.47200789]
drop_score = [0.5514069,  0.58787212, 0.53455358, 0.56154552, 0.4869308 ]
combined_score = [0.54541132, 0.59951113, 0.53779354, 0.57049032, 0.49503245]

plt.boxplot([original_score, drop_score, combined_score],vert=False,showmeans=True)
plt.xlabel('Scores')
plt.ylabel('Model')
plt.title('Model Comparison')

plt.show()


# ### Comment: Model 3 is the best one.

# #### RMSE

# In[57]:


original_rmse = [439363.8769, 418108.6552, 442942.7409, 418938.8163, 496776.7295]
drop_rmse = [426216.8884, 407285.4468, 436178.6955, 411886.2597, 489706.0840]
combined_rmse = [429055.6920, 401493.1292, 434657.9301, 407663.2127, 485824.3327]

plt.boxplot([original_rmse, drop_rmse, combined_rmse],vert=False,showmeans=True)
plt.xlabel('RMSE')
plt.ylabel('Model')
plt.title('RMSE Comparison')

plt.show()


# In[59]:


np.mean(combined_rmse)


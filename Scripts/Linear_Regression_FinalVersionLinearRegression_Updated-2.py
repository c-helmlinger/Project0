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

# In[210]:


df = pd.read_csv("/Users/jessicazhang/Desktop/data.csv")


# In[121]:


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

# In[134]:


# x8 is the building area
# BuildingArea_sqr is the square root.

df['BuildingArea_sqrt'] = np.sqrt(df['BuildingArea'])
df.head()


# ## 4. for x8^2

# In[197]:


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

# In[211]:


# X8 is the building area
# pow() means the square of the number
# i.e. pow(4,2) = 4*4 = 16

df['BuildingArea_sqr'] = pow(df['BuildingArea'],2)
df['Distance_sqr'] = pow(df['Distance'], 2)

df.head()


# # Drop Column

# In[212]:


df = df.drop(['Landsize'], axis=1)
df.head()


# # Spliting into Train and Test Data

# In[213]:


df["YearBuilt"] = 2021 - df["YearBuilt"]
df.head()


# In[214]:


# Drop the last column
X = df.drop(['Price'], axis = 1)


# In[215]:


X


# In[216]:


y = df["Price"]
print(y)


# ## Normalization

# In[217]:


from sklearn import preprocessing


# In[218]:


min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)  


# ## Model

# In[173]:


# Divide the data into two groups: training set and testing set

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=320, test_size=0.27)


# ### K Fold

# In[174]:


from sklearn.model_selection import KFold


# In[219]:


kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[220]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Linear Regression

# In[221]:


model = LinearRegression()


# In[222]:


model.fit(X_train, y_train)


# ## Model Evaluation

# ### Score

# In[21]:


from sklearn.model_selection import cross_val_score


# In[22]:


# (1) Orginal
orginal_score = cross_val_score(model, X_test, y_test, cv=5)
print("Original score:",orginal_score)


# In[208]:


# (2) Drop a column and Add a new column: x8^2
drop_score = cross_val_score(model, X_test, y_test, cv=5)
print("After dropping one column:",drop_score)


# In[223]:


# (3) Drop a column & Feature Engineer (use the combination) 
combined_score = cross_val_score(model, X_test, y_test, cv=5)
print("Combination score:",combined_score)


# ## Based on the Third Model

# ### Coefficients & RMSE

# In[185]:


# y_pred_class = model.predict(X_test)
model.predict(X_test)


# In[186]:


y_test


# In[191]:


result = {'prediction':model.predict(X_test)}


# In[192]:


result_file = pd.DataFrame(result)


# In[193]:


result_file.head()


# In[194]:


result_file.to_csv("Linear_Regression.csv")


# # Result Visualization

# In[195]:


y_predict = result['prediction']
y_test_arr = np.array(list(y_test))
argsort = np.argsort(list(y_test_arr))
sorted_y_test = y_test_arr[argsort]
sorted_y_predict = y_predict[argsort]

index_x = [w for w in range(len(y_predict))]
plt.figure(figsize=(20,10))
plt.plot(index_x, sorted_y_test)

plt.plot(index_x, sorted_y_predict)
plt.legend(['y_true', 'y_predict'])


# ## Model Comparison

# In[224]:


original_score = [0.38793377, 0.24598445, 0.41410552, 0.51712411, 0.46915419]
drop_score = [0.38733962, 0.25136348, 0.41538295, 0.52231204, 0.51973449]
combined_score = [0.40297053, 0.27196173, 0.43784199, 0.55372859, 0.49911831]

plt.boxplot([original_score, drop_score, combined_score],vert=False,showmeans=True)
plt.xlabel('Scores')
plt.ylabel('Model')
plt.title('Model Comparison')

plt.show()


# ### Comment: Model 3 is the best one.

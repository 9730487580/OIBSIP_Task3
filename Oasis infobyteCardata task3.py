#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv("CarData.csv")


# In[ ]:


df


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.CarName.unique()


# In[ ]:


#checking the distributiioin of categorial data
print(df.fueltype.value_counts())
print(df.carbody.value_counts())
print(df.doornumber.value_counts())


# In[ ]:


#encoding the "fuetylpe " column
df.replace({'fueltype':{'Petrol':0,'Diesel':1,'gas':2}},inplace=True)
#encoding the "carbody" column
df.replace({'carbody':{'connvertible':0,'hatchback':1,'sedan':2}},inplace=True)
#encodinng the "doornumber" column
df.replace({'doornumber':{'two':0,'four':1}},inplace=True)


# In[ ]:


df.head()


# In[ ]:


sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(df.price)
plt.show()


# In[ ]:


print(df.corr())


# In[ ]:


plt.figure(figsize=(20, 15))
correlations = df.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[ ]:


predict = "price"
data = df[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]


# In[ ]:


x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[ ]:


from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[257]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[267]:


train_df=pd.read_csv('train.csv')
train_df.head(10)


# In[268]:


train_df.shape


# In[269]:


test_df=pd.read_csv('test.csv')
test_df.head(10)


# In[270]:


train_df.columns


# In[271]:


train_df['SalePrice'].describe()


# In[129]:


sns.distplot(train_df['SalePrice'])


# In[112]:


var='GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
print(data[:5])
data.plot.scatter(x=var,y='SalePrice')


# In[113]:


var = 'TotalBsmtSF'
data=pd.concat([train_df['SalePrice'],train_df[var]],axis=1)
print(data[:5])
data.plot.scatter(x='TotalBsmtSF' ,y='SalePrice')


# In[115]:


var='OverallQual'
data=pd.concat([train_df['SalePrice'],train_df[var]],axis=1)
fig=sns.boxplot(x=var,y='SalePrice',data=data)


# In[116]:


var='YearBuilt'
data=pd.concat([train_df['SalePrice'],train_df[var]],axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


# In[117]:


corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[118]:


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[119]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols],size=2.5)
plt.show()


# In[272]:


total=train_df.isnull().sum().sort_values(ascending=False)
Percent=train_df.isnull().sum()/train_df.isnull().count().sort_values(ascending=False)
missing_data=pd.concat([total,Percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)


# In[273]:


train_df=train_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1)
test_df=test_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1)
train_df.shape


# In[274]:


train_df.head(10)


# In[275]:


total=train_df.isnull().sum().sort_values(ascending=False)
Percent=train_df.isnull().sum()/train_df.isnull().count().sort_values(ascending=False)
missing_data=pd.concat([total,Percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)


# In[276]:


train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)


# In[277]:


train_df.isnull().sum().max()


# In[278]:


saleprice_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[279]:


var='GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
print(data[:5])
data.plot.scatter(x=var,y='SalePrice')


# In[280]:


train_df.sort_values(by = 'GrLivArea', ascending = False)[:2]


# In[281]:


df=train_df['GrLivArea']
print(df.max())


# In[282]:


print(train_df[train_df['Id']==1299])


# In[283]:


print(train_df.loc[1298]['GrLivArea'])


# In[284]:


print(train_df.loc[523]['GrLivArea'])


# In[285]:


train_df=train_df.drop(train_df[train_df['Id']==1299].index)
train_df=train_df.drop(train_df[train_df['Id']==524].index)
train_df.shape


# In[286]:


var='GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
print(data[:5])
data.plot.scatter(x=var,y='SalePrice')


# In[291]:


test=test_df.isnull().sum().sort_values(ascending=False)
test.head(20)


# In[ ]:





# In[293]:


sns.distplot(train_df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)


# In[295]:


train_df['SalePrice'] = np.log(train_df['SalePrice'])


# In[297]:


sns.distplot(train_df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)


# In[298]:


sns.distplot(train_df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['GrLivArea'], plot=plt)


# In[299]:


train_df['GrLivArea'] = np.log(train_df['GrLivArea'])


# In[300]:


sns.distplot(train_df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['GrLivArea'], plot=plt)


# In[304]:





# In[306]:


sns.distplot(train_df['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['TotalBsmtSF'], plot=plt)


# In[310]:


train_df['HasBsmt'] = pd.Series(len(train_df['TotalBsmtSF']), index=train_df.index)
train_df['HasBsmt'] = 0 
train_df.loc[train_df['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[312]:


train_df.loc[train_df['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_df['TotalBsmtSF'])


# In[313]:


sns.distplot(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[314]:


plt.scatter(train_df['GrLivArea'], train_df['SalePrice']);


# In[315]:


plt.scatter(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], train_df[train_df['TotalBsmtSF']>0]['SalePrice']);


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

#read the file from local folder
df_train = pd.read_csv("/Users/yanglulu/Desktop/tdi_rentals_dataset.csv")


# In[92]:


#check to see what the target variable looks like. As shown below, there are 731 values, with a mean value of 4504, a minimum value of 22 and a maximum value of 8714. Based on this observation, we can tell the target variable data is very spread out. 
df_train['rental'].describe()


# In[13]:


# check the distribution of target variable
# looks like it is close to a normal distribution, but with an extra two symmetric peaks. The mean is around 4500.
sns.distplot(df_train['rental'])


# In[18]:


# now let's check occasional and members to see their shapes
sns.distplot(df_train['occasional'])


# In[19]:


sns.distplot(df_train['members'])


# In[20]:


#looks like members and the total rental distributions are similar. Let us check in the same plot
sns.distplot(df_train['rental'])
sns.distplot(df_train['members'])
sns.distplot(df_train['occasional'])


# In[12]:


#check the info about the dataset to see if there is any Null value or if there is any data type that does not make sense.
df_train.info()


# In[14]:


# There is no Null values. The data types look fine. Now we take a look at the summary statistics of all variables
df_train.describe()


# In[15]:


# by the definition of the variables, we can guess some of them are highly correlated, for example: temp and feel_temp, holiday and working day, month and season, weather_condition and humidity and wind_speed. 
# now let us check if our guess is true
corrmat = df_train.corr()
f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[27]:


# our guess was mostly correct! based on the above heatmap we can see the obs and year, month and season, temp and feel_temp are positively correlated, and the working_day and holiday, humidity and wind_speed are negatively correlated.
# In addition, the members and total rentals are highly positively correlated, working_day and occasional, weather condition and all three types of rentals, wind_speed and all three types of rentals are highly negatively correlated. 
# temp and all three rentals, feel_temp and all three rentals, occasional and total rentals, obs and the members/total rentals, year and the members/total rentals, season and the members/total rentals are also positively correlated, and the humidity and all three rental are negatively correlated.
# therefore, we may need to select features to fix the correlation issues between variables. We can use PCA to do that
k=14
cols = corrmat.nlargest(k, 'rental')['rental'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':8},yticklabels=cols.values,xticklabels=cols.values)
plt.show()


# In[28]:


# if we add correlation numbers to the plot, we can see comparing with the feel_temp, temp, year, month and season, the other variables such as wind_speed, humidity, weather conditions , holiday and working_day does not matter much for car rental amounts
# therefore, we can remove the irrelevant variables and train our models, then compare the performance with the model after PCA
# but first let us visualize the relationships of the three most correlated features with the target variable
plt.scatter(df_train['feel_temp'],df_train['rental'])


# In[29]:


plt.scatter(df_train['obs'],df_train['rental'])


# In[30]:


plt.scatter(df_train['temp'],df_train['rental'])


# In[31]:


#then visualize the relationship between occasional and rental, and members and rental
plt.scatter(df_train['occasional'],df_train['rental'])


# In[32]:


plt.scatter(df_train['members'],df_train['rental'])


# In[ ]:


#the relationship are all positively correlated, the obs plot has a bigger variance, the feel_temp plot has some obvious outliers
#the obs plot is interesting because it actually shows how the rental amounts change with time.


# In[146]:


y=df_train['rental']
y.head()


# In[147]:


#for linear regression, all features have to be numerical, also the date column is correlated to obs, year, month and season, so I am dropping date column. 
X=df_train
X=X.drop(['rental','date'],axis=1)
X.head()


# In[148]:


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr=LinearRegression().fit(X_train,y_train)


# In[149]:


print('train_score:{:.2f}'.format(lr.score(X_train,y_train)))
print('test_score:{:.2f}'.format(lr.score(X_test,y_test)))


# In[150]:


#here we found something interesting: both train and test scores are 1. This is unbelievable. We can quickly found the reason, though. Because we included the members and occasional rental amounts in our features, the prediction is undoubtly perfect. Now i am going to remove these two variables and train the model again to see the result.
X=X.drop(['members','occasional'],axis=1)
X.head()


# In[151]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr=LinearRegression().fit(X_train,y_train)
print('train_score:{:.2f}'.format(lr.score(X_train,y_train)))
print('test_score:{:.2f}'.format(lr.score(X_test,y_test)))


# In[152]:


#now we get more reasonable scores!
#however, since we have highly correlated features here in the training model, it may have an overfitting problem. 
#we firstly use cross validation to see if our score is still this high. Then if needed, we will do feature selection to drop some features
from sklearn .model_selection import LeaveOneOut
from sklearn import linear_model
cv=LeaveOneOut()
linear = LinearRegression()

scores=cross_val_score(linear, X,y,cv=20)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[153]:


from sklearn.svm import SVR
for kernel in ['linear','rbf']:
    svr = SVR(kernel=kernel)
    scores = cross_val_score(svr,X,y,cv=20)
    print('iteration:{}'.format(len(scores)))
    print('average score:{}'.format(scores.mean()))


# In[ ]:


#The results are not satisfying. There must be something wrong with the variables (multicollinearity or outliers). We need to figure it out


# In[154]:


# or, let us regress on the two supporting variables : occasional and members. Then regress these two on the target variable rental
X_members = X
y_members = df_train['members']
X_occasional = X
y_occasional = df_train['occasional']

scores=cross_val_score(linear, X_members,y_members,cv=20)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[155]:


scores=cross_val_score(linear, X_occasional,y_occasional,cv=20)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[157]:


#based on above experiment, we can guess that the main reason that the model was not working to predict 'rental' is because of the 'occasional' part. 
#Let us do some feature selection
X_occasional = X_occasional.drop(['temp','holiday','humidity'],axis=1)
scores=cross_val_score(linear, X_occasional,y_occasional,cv=20)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[158]:


X_occasional = X_occasional.drop(['month'],axis=1)
scores=cross_val_score(linear, X_occasional,y_occasional,cv=20)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[159]:


X_occasional = X_occasional.drop(['obs'],axis=1)
scores=cross_val_score(linear, X_occasional,y_occasional,cv=20)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[136]:


#this result is not satisfying. There must be some problems with our feature that are causing nan
X=X.drop(['obs'],axis=1)
X.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[137]:


scores=cross_val_score(linear, X,y,cv=5)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[138]:


X=X.drop(['feel_temp'],axis=1)
X.head()


# In[139]:


scores=cross_val_score(linear, X,y,cv=5)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[128]:


X=X.drop(['holiday'],axis=1)
X.head()


# In[140]:


scores=cross_val_score(linear, X,y,cv=10)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[141]:


rf=RandomForestRegressor()
scores=cross_val_score(rf, X,y,cv=10)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[142]:


dt=DecisionTreeRegressor()
scores=cross_val_score(dt, X,y,cv=10)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[143]:


log = LogisticRegression()
scores=cross_val_score(log, X,y,cv=10)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


# In[145]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import xgboost as xgb
models = []
#models.append(("LR",LogisticRegression()))
#models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestRegressor()))
#models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeRegressor()))
#models.append(("XGB",xgb.XGBClassifier()))
#models.append(("KNN",KNeighborsClassifier()))

for name,model in models:
   cv_result = cross_val_score(model,X,y)
   print(name, cv_result)


# In[160]:


sns.set()
cols = df_train.columns
sns.pairplot(df_train[cols],size=2.5)
plt.show()


# In[ ]:


scores=cross_val_score(linear, df_train[],y,cv=10)
print('iteration:{}'.format(len(scores)))
print('average score:{}'.format(scores.mean()))


#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # Lets load the Boston House Pricing Dataset

# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston=load_boston()


# In[4]:


boston.keys()


# In[5]:


## Lets check the description of the dataset
print(boston.DESCR)


# In[6]:


print(boston.data)


# In[7]:


print(boston.target)


# In[8]:


print(boston.feature_names)


# # Preparing The Dataset

# In[9]:


dataset=pd.DataFrame(boston.data,columns=boston.feature_names) #All features value with Column name(Bosten.data not contain target variable)


# In[10]:


dataset.head()


# In[11]:


dataset['Price']=boston.target #Target Variable


# In[12]:


dataset.head() #Complete Dataset


# # EDA

# In[13]:


dataset.info() #Dataset information


# In[14]:


# Summarizing The Stats of the data (Only for numarical values)
dataset.describe()


# In[15]:


# Check the missing Values 
dataset.isnull().sum()


# In[16]:


# Correlation (If our independent features are highly correlated(positively or negatively) with dependent variable this indicate that our model performance is high )
dataset.corr() # (If there is correlation between independent features then we can remove on of them\ Multicollinearity(features are highly correlated))


# In[17]:


# Analyze Correlation with plots


# In[18]:


import seaborn as sns
sns.pairplot(dataset)


# # Analyzing The Correlated Features

# In[19]:


dataset.corr()


# In[20]:


plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")
# There is a relationship between crime rate and price(if Crime rate increasing then price decreses) )


# In[21]:


plt.scatter(dataset['RM'],dataset['Price'])
plt.xlabel("RM") #average number of rooms per dwelling
plt.ylabel("Price")


# In[22]:


import seaborn as sns
sns.regplot(x="RM",y="Price",data=dataset)


# In[23]:


sns.regplot(x="LSTAT",y="Price",data=dataset) #LSTAT=  % lower status of the population


# In[24]:


sns.regplot(x="CHAS",y="Price",data=dataset) # No Correlation
#Linearity should be in your dataset to create a best regression model
# if there is a features which are hardly correlated it reduces the error of the regression model(?)


# In[25]:


sns.regplot(x="PTRATIO",y="Price",data=dataset)


# In[26]:


#Create a multiple linear regression model


# In[13]:


## Independent and Dependent features
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[14]:


X.head()


# In[15]:


y


# In[16]:


##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[17]:


X_train


# In[18]:


X_test


# In[19]:


# In gradient decent our target is to reach globle minima and here all features are calculated with respect to different units
# Therefor to converge globel minima fastly we have to normalize to all these data points to convert it on same scale


# In[20]:


## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[21]:


X_train=scaler.fit_transform(X_train)


# In[22]:


X_test=scaler.transform(X_test) #Model dont know much about the test dataset


# In[24]:


X_train


# In[25]:


X_test


# # Model Training

# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score # Every combenation of train and test data are taken by model and whos accuracy is more its checked. 


# In[27]:


regression=LinearRegression()


# In[28]:


regression.fit(X_train,y_train)


# In[29]:


# print the coefficients and the intercept
print(regression.coef_)  #Change in y wrt unit change in x values


# In[30]:


print(regression.intercept_) #expected value of Y when all X=0


# In[31]:


# on which parameters the model has been trained
regression.get_params()


# In[32]:


# Prediction With Test Data
reg_pred=regression.predict(X_test)


# In[33]:


reg_pred


# In[34]:


mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[45]:


# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()

params={'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,50,100]} #Dictionary #hyperparameter

ridge_regressor=GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[49]:


print(ridge_regressor.best_params_) #Best value for alpha in model
print(ridge_regressor.best_score_) #Best mse score according to that alpha


# In[47]:


# Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()

params={'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,50,100]} #Dictionary #hyperparameter

lasso_regressor=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train,y_train)


# In[48]:


print(lasso_regressor.best_params_) #Best value for alpha in model
print(lasso_regressor.best_score_) #Best mse score according to that alpha


# In[54]:


#Prediction through lasso and Ridge
y_pred_lasso=lasso_regressor.predict(X_test)
y_pred_ridge=ridge_regressor.predict(X_test)

from sklearn.metrics import r2_score
r_square_lasso=r2_score(y_pred_lasso,y_test)
r_square_ridge=r2_score(y_pred_ridge,y_test)
print(r_square_lasso)
print(r_square_ridge)


# # Assumptions

# In[45]:


# plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred) #Prediction plot and here we see that our model performing well


# In[46]:


# Residuals
residuals=y_test-reg_pred


# In[47]:


residuals


# In[48]:


# Plot this residuals 
# Our assumption for linear regression is errors are normaly distributed
# Residuals are representation of error
sns.displot(residuals,kind="kde") # from graph we see that its normal but some outliers are present


# In[49]:


# Scatter plot with respect to prediction and residuals 
# uniform distribution
plt.scatter(reg_pred,residuals)


# In[44]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# # R square and adjusted R square
# 
# Formula
# 
# R^2 = 1 - SSR/SST
# 
# R^2 = coefficient of determination SSR = sum of squares of residuals SST = total sum of squares
# 

# In[51]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]
# 
# where:
# 
# R2: The R2 of the model n: The number of observations k: The number of predictor variables

# In[52]:


#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# # New Data Prediction

# In[53]:


boston.data[0].reshape(1,-1)


# In[54]:


##transformation of new data
scaler.transform(boston.data[0].reshape(1,-1))


# In[55]:


regression.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# In[ ]:





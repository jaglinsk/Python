#!/usr/bin/env python
# coding: utf-8

# The following code imports and analyzes facebook advertising data, with the goal of developing a predictive model for ad conversions

# In[1]:


import pandas as pd
import math
import os
import sys
from pathlib import Path
import csv
import seaborn as sn
import numpy as np

from matplotlib import pyplot

import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[2]:


os.chdir('C:/Users/Bridget Marie Yoga/Documents/GitPy/Python')


# Due to file being non-ideal (missing values, trailing commas in some rows), it is best to clean up the data before importing it into our enviornment

# In[3]:


filename = Path('data.csv')
newname = filename.parent/f"{filename.stem}-fixed{filename.suffix}"

BADCOLS = ['', '']

with open(filename, newline='') as infile, open(newname, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for row in csv.reader(infile):
        if row[-2:] == BADCOLS:
            row[3:3] = BADCOLS
            del row[-2:]
        writer.writerow(row)

# test it

ads = pd.read_csv(newname, header=None)
print(ads)


# In[4]:


#Rename columns
ads.columns = ads.iloc[0]
ads = ads.drop(0)

ads.head()
print(ads.dtypes) 


# In[5]:


#Adjust type for columns via apply()
ads[['impressions', 'clicks','spent','total_conversion', 'approved_conversion']] = ads[['impressions', 'clicks','spent','total_conversion', 'approved_conversion']].apply(pd.to_numeric) 
ads['reporting_start'] = pd.to_datetime(ads['reporting_start'])
ads['reporting_end'] = pd.to_datetime(ads['reporting_end'])

ads['day_of_week'] = ads['reporting_start'].dt.day_name()


# In[6]:


#Create custom marketing values
ads['cost_per_click'] = ads['spent'] / ads['clicks']

#df['day_of_week'] = df['my_dates'].dt.day_name()

#ads['cost_per_acquisition'] =ads['spent'].astype('int32') / ads['approved_conversion'].astype('int32')


# Exploratory data analysis

# In[7]:


ads.describe()


# In[8]:


impressions_by_gender = pd.pivot_table(ads,index = 'gender', columns = 'age', values = 'impressions',aggfunc = np.count_nonzero)
impressions_by_gender.plot.bar()
print(impressions_by_gender)


# In[9]:


conversions_by_campaign = pd.pivot_table(ads,index = 'campaign_id', values = 'approved_conversion',aggfunc = np.sum)
conversions_by_campaign.plot.bar()
print(conversions_by_campaign)


# In[10]:


corrMatrix = ads.corr()
sn.heatmap(corrMatrix, annot=True)


# Create model pipelines for: Data imputation, scaling, and best fit model identificatio 

# In[11]:


def label_convert(row):
    if row['approved_conversion'] == 0:
        return 0
    else:
        return 1


# In[12]:


ads['true_convert'] = ads.apply (lambda row: label_convert(row), axis=1)


# In[13]:


#Removed certain columns based on previous iterations to hone in on characteristics we can control in our reporting strategy
x = ads.drop(columns=['approved_conversion','total_conversion','reporting_start','reporting_end', 'ad_id','fb_campaign_id','true_convert','cost_per_click'])
y = ads['true_convert']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.33, random_state=85)


# In[14]:


x.columns


# In[15]:


y_train.value_counts()


# In[16]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[17]:


numeric_features = ads.select_dtypes(include=['int64', 'float64']).drop(['approved_conversion','total_conversion', 'true_convert','cost_per_click'], axis=1).columns
categorical_features = ads.select_dtypes(include=['object']).drop([ 'ad_id','fb_campaign_id'],axis=1).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# In[18]:


numeric_features


# In[19]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(random_state=42)]
for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
    pipe.fit(x_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipe.score(x_test, y_test))


# With the best model identified, leverage GridSearchCV() to optimize hyperparameters

# In[20]:


scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),'recall':make_scorer(recall_score)}

param_grid = [{
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }]

n_cpus = mp.cpu_count()
#Does this make sense with scoring and refit different scores?
gridsearch = GridSearchCV(GradientBoostingClassifier(), param_grid, n_jobs= n_cpus,verbose=0.5, scoring='f1',refit='accuracy_score',cv=2)


# In[21]:


pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('gridsearch', gridsearch)])
pipe.fit(x_train, y_train)
#print(pipe.best_params_)    
#print(pipe.best_score_)


# In[22]:


pipe[1]


# Identify key feature importances of facebook ad campaigns

# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

importance = gridsearch.best_estimator_.feature_importances_

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train.columns, importance):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)


# In[ ]:


print(importances)


# In[ ]:





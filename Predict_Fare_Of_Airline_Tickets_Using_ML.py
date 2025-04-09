#!/usr/bin/env python
# coding: utf-8

# In[5]:


## import necessary packages !

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# importing Dataset

train_data = pd.read_excel(r"C:\Users\HP ECS\Downloads\Flight_Price_resources\Data_Train.xlsx")


# In[7]:


train_data.head(4)


# In[8]:


train_data.tail(4)


# In[9]:


train_data.info()


# In[10]:


train_data.isnull().sum()


# In[11]:


train_data['Total_Stops'].isnull()


# In[12]:


# Getting all the rows where we have missing value

train_data[train_data['Total_Stops'].isnull()]


# In[13]:


# We have only one missing value so we can Drop that one missing value

train_data.dropna(inplace=True)


# In[14]:


train_data.isnull().sum()


# In[15]:


train_data.dtypes


# In[16]:


train_data.info(memory_usage="deep")


# In[17]:


# Performed Data pre-processing and extract derived attributes from "Date_of_Journey"

data = train_data.copy()


# In[18]:


data.columns


# In[19]:


data.head(2)


# In[20]:


data.dtypes


# In[21]:


def change_into_Datetime(col):
    data[col] = pd.to_datetime(data[col])


# In[22]:


import warnings 
from warnings import filterwarnings
filterwarnings("ignore")


# In[23]:


data.columns


# In[24]:


for feature in ['Dep_Time', 'Arrival_Time' , 'Date_of_Journey']:
    change_into_Datetime(feature)


# In[25]:


data.dtypes


# In[26]:


data["Journey_day"] = data['Date_of_Journey'].dt.day


# In[27]:


data["Journey_month"] = data['Date_of_Journey'].dt.month


# In[28]:


data["Journey_year"] = data['Date_of_Journey'].dt.year


# In[29]:


data.head(3)


# In[30]:


# Clean Departure time and Arrival time and then extract derived attributes

def extract_hour_min(df , col):
    df[col+"_hour"] = df[col].dt.hour
    df[col+"_minute"] = df[col].dt.minute
    return df.head(3)


# In[31]:


data.columns


# In[32]:


# Departure time is when a plane leaves the gate. 

extract_hour_min(data , "Dep_Time")


# In[33]:


extract_hour_min(data , "Arrival_Time")


# In[34]:


## we have extracted derived attributes from ['Arrival_Time' , "Dep_Time"] , so lets drop both these features ..
cols_to_drop = ['Arrival_Time' , "Dep_Time"]

data.drop(cols_to_drop , axis=1 , inplace=True )


# In[35]:


data.head(3)


# In[36]:


data.shape


# In[37]:


# Lets analyse when will most of the flight take off

data.columns


# In[38]:


#### Converting the flight Dep_Time into proper time i.e. mid_night, morning, afternoon and evening.

def flight_dep_time(x):
    '''
    This function takes the flight Departure time 
    and convert into appropriate format.
    
    '''
    
    if (x>4) and (x<=8):
        return "Early Morning"
    
    elif (x>8) and (x<=12):
        return "Morning"
    
    elif (x>12) and (x<=16):
        return "Noon"
    
    elif (x>16) and (x<=20):
        return "Evening"
    
    elif (x>20) and (x<=24):
        return "Night"
    
    else:
        return "late night"


# In[39]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind="bar" , color="g")


# In[40]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot , iplot , init_notebook_mode , download_plotlyjs
init_notebook_mode(connected=True)
cf.go_offline()


# In[41]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind="bar")


# In[42]:


# Pre-processing on duration feature and extract meaningful features from it

data.head(3)


# In[43]:


def preprocess_duration(x):
    if 'h' not in x:
        x = '0h' + ' ' + x
    elif 'm' not in x:
        x = x + ' ' +'0m'
        
    return x


# In[44]:


data['Duration'] = data['Duration'].apply(preprocess_duration)


# In[45]:


data['Duration']


# In[46]:


data['Duration'][0]


# In[47]:


'2h 50m'.split(' ')


# In[48]:


'2h 50m'.split(' ')[0]


# In[49]:


'2h 50m'.split(' ')[0][0:-1]


# In[50]:


type('2h 50m'.split(' ')[0][0:-1])


# In[51]:


int('2h 50m'.split(' ')[0][0:-1])


# In[52]:


int('2h 50m'.split(' ')[1][0:-1])


# In[53]:


data['Duration_hours'] = data['Duration'].apply(lambda x : int(x.split(' ')[0][0:-1]))


# In[54]:


data['Duration_mins'] = data['Duration'].apply(lambda x : int(x.split(' ')[1][0:-1]))


# In[55]:


data.head(2)


# In[56]:


pd.to_timedelta(data["Duration"]).dt.components.hours


# In[57]:


data["Duration_hour"] = pd.to_timedelta(data["Duration"]).dt.components.hours


# In[58]:


data["Duration_minute"] = pd.to_timedelta(data["Duration"]).dt.components.minutes


# In[59]:


# Analyse weather duration impacts price or not

data['Duration'] ## convert duration into total minutes duration ..


# In[60]:


2*60


# In[61]:


'2*60'


# In[62]:


eval('2*60')


# In[63]:


data['Duration_total_mins'] = data['Duration'].str.replace('h' ,"*60").str.replace(' ' , '+').str.replace('m' , "*1").apply(eval)


# In[64]:


data['Duration_total_mins']


# In[65]:


data.columns


# In[66]:


sns.scatterplot(x="Duration_total_mins" , y="Price" , data=data)


# In[67]:


### As the duration of minutes increases Flight price also increases.

sns.lmplot(x="Duration_total_mins" , y="Price" , data=data)


# In[68]:


### lets understand whether total stops affect price or not !

sns.scatterplot(x="Duration_total_mins" , y="Price" , hue="Total_Stops", data=data)


# In[69]:


# Analyse on which route jet Airways is extreamly used?

data['Airline']=='Jet Airways'


# In[70]:


data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[71]:


data.columns


# In[72]:


# Performing Airline vs Price Analysis

sns.boxplot(y='Price' , x='Airline' , data=data.sort_values('Price' , ascending=False))
plt.xticks(rotation="vertical")
plt.show()


# In[73]:


# Apply one-hot Encoading on data

data.head(2)


# In[74]:


cat_col = [col for col in data.columns if data[col].dtype=="object"]


# In[75]:


num_col = [col for col in data.columns if data[col].dtype!="object"]


# In[76]:


cat_col


# In[77]:


### Applying One-hot from scratch :

data['Source'].unique()


# In[78]:


data['Source'].apply(lambda x : 1 if x=='Banglore' else 0)


# In[79]:


for sub_category in data['Source'].unique():
    data['Source_'+sub_category] = data['Source'].apply(lambda x : 1 if x==sub_category else 0)


# In[80]:


data.head(3)


# In[81]:


# Handling Categorical Data

cat_col


# In[82]:


data.head(2)


# In[83]:


data['Airline'].nunique()


# In[84]:


data.groupby(['Airline'])['Price'].mean().sort_values()


# In[85]:


airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[86]:


airlines


# In[87]:


dict_airlines = {key:index for index , key in enumerate(airlines , 0)}


# In[88]:


dict_airlines


# In[89]:


data['Airline'] = data['Airline'].map(dict_airlines)


# In[90]:


data['Airline']


# In[91]:


data.head(3)


# In[92]:


data['Destination'].unique()


# In[93]:


data['Destination'].replace('New Delhi' , 'Delhi' , inplace=True)


# In[94]:


data['Destination'].unique()


# In[95]:


dest = data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[96]:


dest


# In[97]:


dict_dest = {key:index for index , key in enumerate(dest , 0)}


# In[98]:


dict_dest


# In[99]:


data['Destination'] = data['Destination'].map(dict_dest)


# In[100]:


data['Destination']


# In[101]:


data.head(3)


# In[102]:


# Perform label(manual) Encoading on data

data.head(3)


# In[103]:


data['Total_Stops']


# In[104]:


data['Total_Stops'].unique()


# In[105]:


stop = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[106]:


data['Total_Stops'] = data['Total_Stops'].map(stop)


# In[107]:


data['Total_Stops']


# In[108]:


# Remove Un-necessary Features

data.head(1)


# In[109]:


data.columns


# In[110]:


data['Additional_Info'].value_counts()/len(data)*100


# In[111]:


data.head(4)


# In[112]:


data.columns


# In[113]:


data['Journey_year'].unique()


# In[114]:


data.drop(columns=['Date_of_Journey' , 'Additional_Info' , 'Duration_total_mins' , 'Source' , 'Journey_year'] , axis=1 , inplace=True)


# In[115]:


data.columns


# In[116]:


data.head(4)


# In[117]:


data.drop(columns=['Route'] , axis=1 , inplace=True)


# In[118]:


data.head(3)


# In[119]:


data.drop(columns=['Duration'] , axis=1 , inplace=True)


# In[120]:


data.head(3)


# In[121]:


# Perform Outlier Detection

def plot(df, col):
    fig , (ax1 , ax2 , ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col] , ax=ax1)
    sns.boxplot(df[col] , ax=ax2)
    sns.distplot(df[col] , ax=ax3 , kde=False)


# In[122]:


plot(data , 'Price')


# In[123]:


q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3- q1

maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr


# In[124]:


print(maximum)


# In[125]:


print(minimum)


# In[126]:


print([price for price in data['Price'] if price> maximum or price<minimum])


# In[127]:


len([price for price in data['Price'] if price> maximum or price<minimum])


# In[128]:


### wherever I have price >35K just replace replace it with median of Price

data['Price'] = np.where(data['Price']>=35000 , data['Price'].median() , data['Price'])


# In[129]:


plot(data , 'Price')


# In[130]:


# Perform Feature Selection

X = data.drop(['Price'] , axis=1)


# In[131]:


y = data['Price']


# In[132]:


from sklearn.feature_selection import mutual_info_regression


# In[133]:


imp = mutual_info_regression(X , y)


# In[134]:


imp


# In[135]:


imp_df = pd.DataFrame(imp , index=X.columns)


# In[136]:


imp_df.columns = ['importance']


# In[137]:


imp_df


# In[138]:


imp_df.sort_values(by='importance' , ascending=False)


# In[139]:


# Lets build ML Model

from sklearn.model_selection import train_test_split


# In[140]:


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.25, random_state=42)


# In[141]:


from sklearn.ensemble import RandomForestRegressor


# In[142]:


ml_model = RandomForestRegressor()


# In[143]:


ml_model.fit(X_train,y_train)


# In[144]:


y_pred = ml_model.predict(X_test)


# In[145]:


y_pred


# In[146]:


from sklearn import metrics


# In[147]:


metrics.r2_score(y_test , y_pred)


# In[148]:


# How to automate ML Pipelines and How to define Evaluation Metrics

def mape(y_true , y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[149]:


mape(y_test , y_pred)


# In[150]:


from sklearn import metrics


# In[151]:


def predict(ml_model):
    model = ml_model.fit(X_train , y_train)
    print('Training score : {}'.format(model.score(X_train , y_train)))
    y_predection = model.predict(X_test)
    print('predictions are : {}'.format(y_predection))
    print('\n')
    r2_score = metrics.r2_score(y_test , y_predection)
    print('r2 score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test , y_predection)))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test , y_predection)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test , y_predection))))
    print('MAPE : {}'.format(mape(y_test , y_predection)))
    sns.distplot(y_test - y_predection)


# In[152]:


predict(RandomForestRegressor())


# In[153]:


from sklearn.tree import DecisionTreeRegressor


# In[154]:


predict(DecisionTreeRegressor())


# In[155]:


# How to hyper-tune ML model

from sklearn.model_selection import RandomizedSearchCV


# In[156]:


### initialise your estimator
reg_rf = RandomForestRegressor()


# In[157]:


np.linspace(start =100 , stop=1200 , num=6)


# In[158]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start =100 , stop=1200 , num=6)]

# Number of features to consider at every split
max_features = ["auto", "sqrt"]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start =5 , stop=30 , num=4)]

# Minimum number of samples required to split a node
min_samples_split = [5,10,15,100]


# In[159]:


# Create the random grid or hyper-parameter space

random_grid = {
    'n_estimators' : n_estimators , 
    'max_features' : max_features , 
    'max_depth' : max_depth , 
    'min_samples_split' : min_samples_split
}


# In[160]:


random_grid


# In[161]:


## Define searching

# Random search of parameters, using 3 fold cross validation
# search across 576 different combinations


rf_random = RandomizedSearchCV(estimator=reg_rf , param_distributions=random_grid , cv=3 , n_jobs=-1 , verbose=2)


# In[162]:


rf_random.fit(X_train , y_train)


# In[163]:


rf_random.best_params_


# In[164]:


rf_random.best_estimator_


# In[165]:


rf_random.best_score_


# In[ ]:





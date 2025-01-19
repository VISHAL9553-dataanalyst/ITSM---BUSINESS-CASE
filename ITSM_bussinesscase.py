#!/usr/bin/env python
# coding: utf-8

# !pip install pymysql
# !pip install mysql-connector

# # Project Ref: PM-PR-0012 Client: ABC Tech | Category: ITSM - ML

# ## Team ID : PTID-CDS-JUL-23-1658

# # Business Case:
# ABC Tech is an mid-size organisation operation in IT-enabled business segment over a decade. On an average ABC Tech receives 22-25k IT incidents/tickets , which were handled to best practice ITIL framework with incident management , problem management, change management and configuration management processes. These ITIL practices attained matured process level and a recent audit confirmed that further improvement initiatives may not yield return of investment. ABC Tech management is looking for ways to improve the incident management process as recent customer survey results shows that incident management is rated as poor. Machine Learning as way to improve ITSM processes ABC Tech management recently attended Machine Learning conference on ML for ITSM. Machine learning looks prospective to improve ITSM processes through prediction and automation. They came up with 4 key areas, where ML can help ITSM process in ABC Tech.
# 1. Predicting High Priority Tickets: To predict priority 1 & 2 tickets, so that they can take preventive measures or fix the problem before
# it surfaces.
# 2. Forecast the incident volume in different fields , quarterly and annual. So that they can be better prepared with resources and technology planning.
# 3. Auto tag the tickets with right priorities and right departments so that reassigning and related delay can be reduced.
# 4. Predict RFC (Request for change) and possible failure / 
# misconfiguration of ITSM assets
# # Data Set Fields:2 
# Total of about 46k records from year 2012,2013,201 4
# Data needs to be queried from MYSQL data base (Read Only Acces s)
# Host: 18.136.157. 135
# Port:
#  3306
# Username : dm _team
# Password: DM!$Team@27920!
# .

# # Import Required Libraries

# In[1]:


import os
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime as dt
import itertools
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings("ignore")


# # Data Collection

# In[2]:


bold="\033[1m"
reset="\033[0m"
red="\033[4;31m"
cyan="\033[4;36m"
green="\033[4;32m"
purple="\033[4;35m"


# In[3]:


connection=mysql.connector.connect(host = '18.136.157.135',
                                  user='dm_team',
                                  password='DM!$Team@&27920!',
                                  database='project_itsm')


# In[4]:


cursor=connection.cursor()
cursor.execute('show databases')
for i in cursor:
    print(i)


# In[5]:


data_tables=pd.read_sql_query("show tables",connection)
print(data_tables)


# In[6]:


print(bold+purple+"Load dataset:-"+reset)
data=pd.read_sql_query("select*from dataset_list",connection)
pd.set_option('display.max_columns',None)
print(bold+red+"Shape of dataset:"+reset,data.shape)
print(bold+green+"Size of dataset:"+reset,data.size)
data.head(50)


# In[7]:


data.info()
print(bold+green+"All data features are in form of object datatypes.")


# In[8]:


print(bold+red+"Total high priority of tickets count"+reset)
query2="select * from dataset_list where priority in ('1','2') order by priority desc"
q2=pd.read_sql(query2,connection)
q2


# In[9]:


print(bold+purple+"Delve of frequently occuring incident in subcategory on basis of priority.")
query3="select CI_Cat,CI_Subcat,Priority,Category,No_of_Reassignments,No_of_Related_Incidents,Open_Time,Reopen_Time from dataset_list where Category='incident'"
q3=pd.read_sql_query(query3,connection)
pd.set_option("display.max_rows",None)
q3


# # Data Preprocessing

# ## Duplicate data processing

# In[10]:


dup_values=data.duplicated().sum()
print(bold+green+"Not have any duplicated value availble in our datasets:-"+reset,dup_values)


# ## Drop data 

# In[11]:


print(bold+purple+"Drop data which does not useful as per domain kowledge.")
data.drop(["CI_Name","Status","number_cnt","KB_number","Alert_Status","Resolved_Time","Close_Time","Handle_Time_hrs","Related_Interaction",
         "Related_Change","Closure_Code"],axis=1,inplace=True)


# In[12]:


data.head(1)


# ## Fix Missing value and datatype conversion

# #### CI_Cat features

# In[13]:


data.CI_Cat.unique()


# In[14]:


Missing_data=data.loc[data["CI_Cat"]=='']
print(bold+green+"Total missing data available in ' ' length:-",len(Missing_data))


# In[15]:


print(bold+red+"Replace records unto common records and change datatypes.")
hardware_device=('computer','hardware','displaydevice','officeelectronics','Phone')
database=('storage','database')
Others=('')
data["CI_Cat"]=data["CI_Cat"].replace(Others,"others").astype('category')
data["CI_Cat"]=data["CI_Cat"].replace(hardware_device,"hardware").astype('category')
data["CI_Cat"]=data["CI_Cat"].replace(database,"database").astype('category')


# In[16]:


# '' missing data convert into others form.
data.loc[data["CI_Cat"]=='']


# In[17]:


plt.figure(figsize=(20,20))
dp=sns.countplot(data,x="CI_Cat",palette='seismic')
for i in dp.patches:
    x=i.get_x()+i.get_width()/2
    y=i.get_height()
    dp.annotate(f"{y:.0f}",(x,y),ha='center',fontsize=10)
plt.xlabel('CI_Cat',fontsize=20)
plt.ylabel('count',fontsize=20)
plt.show()


# #### CI_Subcat features

# In[17]:


subcat=data.CI_Subcat.unique()
print(bold+green+"Total category:\n"+reset,subcat)


# In[18]:


print(bold+purple+"We are found that missing '' values based records in CI_Subcat features. this is include in other category.")


# In[19]:


hardware_cat=('Laptop','Monitor','Desktop','Banking Device','Keyboard','Printer','Scanner','NonStop Storage','Tape Library','VDI','Controller','NonStop Harddisk','DataCenterEquipment','KVM Switches','Switch')
internet_sof=('Exchange','Automation Software','Client Based Application','Encryption','Security Software','System Software','Iptelephony','Web Based Application','Desktop Application','Citrix','Standard Application','IPtelephony','zOS Systeem','Database Software','MigratieDummy','SharePoint Farm','SAP','Firewall')
server=('NonStop Server','Windows Server in extern beheer','Windows Server','Linux Server','Oracle Server','Unix Server','ESX Server','Application Server','Thin Client','Server Based Application','Neoview Server','X86 Server','Virtual Tape Server','zOS Server')
networking=('Router','Protocol','Modem','Network Component','Net Device','zOS Cluster','VMWare','Lines','ESX Cluster','MQ Queue Manager','Instance')
others=('Number','RAC Service','Omgeving','SAN','UPS','Database','')
data['CI_Subcat']=data['CI_Subcat'].replace(hardware_cat,"hardware").astype('category')
data['CI_Subcat']=data['CI_Subcat'].replace(internet_sof,"internet_software").astype('category')
data['CI_Subcat']=data['CI_Subcat'].replace(server,"server").astype('category')
data['CI_Subcat']=data['CI_Subcat'].replace(networking,"networking").astype('category')
data['CI_Subcat']=data['CI_Subcat'].replace(others,"others").astype('category')


# In[20]:


data.CI_Subcat.value_counts()


# In[21]:


print(bold+cyan+"Check others features category"+reset)
data.loc[data["CI_Subcat"]=="others"]


# In[23]:


plt.figure(figsize=(20,20))
dp=sns.countplot(data,x="CI_Subcat",palette='icefire')
for i in dp.patches:
    x=i.get_x()+i.get_width()/2
    y=i.get_height()
    dp.annotate(f"{y:.0f}",(x,y),ha='center',fontsize=10)
plt.xlabel('CI_Subcat',fontsize=20)
plt.ylabel('count',fontsize=20)
plt.show()


# #### Impact features

# In[22]:


data.Impact.value_counts()


# In[23]:


print(bold+green+"Convert categorical value into numerical values.")
data["Impact"]=data["Impact"].replace("NS",0).astype('int')


# In[24]:


print(bold+cyan+"Verified total category in Impact features.")
data.Impact.unique()


# #### Urgency features

# In[25]:


data.Urgency.value_counts()


# In[26]:


data.drop(data[data["Urgency"]=='5 - Very Low'].index,inplace=True)


# In[27]:


data.Urgency.unique()


# In[28]:


print(bold+cyan+"convert datatype into integer.")
data["Urgency"]=data["Urgency"].astype(int)


# #### Priority features

# In[29]:


data['Priority'].unique()


# In[30]:


data['Priority'].value_counts()


# In[31]:


print(bold+green+"Convert categorical value convert into numerical form with replace method.")
data['Priority']=data['Priority'].replace('NA',6).astype(int)


# In[32]:


print(bold+red+"check datatyped of priority features"+reset)
print(data['Priority'].dtype)


# In[33]:


print(bold+green+"HIgh priority convert into 1 and low priority convert into 0.")
data['Priority']=data['Priority'].apply(lambda x: 1 if x in [1,2] else 0)


# In[34]:


data.Priority.value_counts()


# #### No_of_Reassignments

# In[35]:


print(bold+cyan+"Check unique value of No_of_Ressignment.")
data.No_of_Reassignments.unique()


# In[36]:


print(bold+purple+"Missing value dropped.")
data.drop(data[data["No_of_Reassignments"]==''].index,inplace=True)


# In[37]:


print(bold+red+"Change datatype form object to int."+reset)
data["No_of_Reassignments"]=data["No_of_Reassignments"].astype(int)


# In[38]:


print(bold+green+"Reverified unique values of No_of_Reassignments features.")
data.No_of_Reassignments.unique()


# #### Reopen time

# In[39]:


data.Reopen_Time.unique()


# In[40]:


data["Reopen_time"]=pd.to_datetime(data["Reopen_Time"],format='%d-%m-%Y %H:%M',errors='coerce').dt.date
data["Reopen_time"].fillna(0,inplace=True)
data.drop('Reopen_Time',axis=1,inplace=True)


# In[41]:


print(bold+red+"Convert object function into datetime format.")
data['Reopen_time']=pd.to_datetime(data['Reopen_time'])


# In[42]:


data.head(2)


# #### Open_Time

# In[43]:


data.Open_Time.unique()


# In[44]:


data["open_time"]=pd.to_datetime(data["Open_Time"],format='%d-%m-%Y %H:%M',errors='coerce').dt.date
data.drop("Open_Time",axis=1,inplace=True)


# In[45]:


print(bold+cyan+"Convert object datatype into datetime format.")
data['open_time']=pd.to_datetime(data['open_time'])


# In[46]:


data.head(2)


# #### No_of_Related_Interactions

# In[47]:


data.No_of_Related_Interactions.unique()


# In[48]:


print(bold+purple+"Find out one missing value"+reset)
data.No_of_Related_Interactions.value_counts()


# In[49]:


print(bold+cyan+"This missing value transfor into 1 no of interactions category beacause majority values lies in it."+reset)
data["No_of_Related_Interactions"]=data["No_of_Related_Interactions"].replace('',1).astype(int)


# In[50]:


print(bold+red+"Verified total changes of dataset.")
data["No_of_Related_Interactions"].unique()


# #### No_of_Related_Incidents 

# In[51]:


data.No_of_Related_Incidents.unique()


# In[52]:


print(bold+purple+"Missing value fix in this features."+reset)
data["No_of_Related_Incidents"]=data["No_of_Related_Incidents"].replace('',1).astype(int)


# In[53]:


print(bold+green+"Check out missing values in related incident features.")
data.No_of_Related_Incidents.value_counts()


# #### No_of_Related_Changes

# In[54]:


data.No_of_Related_Changes.unique()


# In[55]:


print(bold+cyan+"No of related changes features are lot of missing value available."+reset)
data.No_of_Related_Changes.value_counts()


# In[56]:


print(bold+red+"Fix the missing value in this features."+reset)
data["No_of_Related_Changes"]=data["No_of_Related_Changes"].replace('',1).astype(int)


# In[57]:


data.No_of_Related_Changes.value_counts()


# In[58]:


print(bold+green+'Object category convert into categorical column.')
datatype=['WBS','Incident_ID','Category']
for i in datatype:
    data[i]=data[i].astype('category')


# In[59]:


data.info()


# ## Encoding process

# In[60]:


# convert categorical data into int format with help of label encoder method.
le=LabelEncoder()
categorical_data=data.select_dtypes(include='category').columns
data[categorical_data]=data[categorical_data].apply(lambda col: le.fit_transform(col))
data.head()


# ## Outliers Method 

# In[79]:


plt.figure(figsize=(10,10))
plotnumber=1
for i in data:
    if plotnumber<12:
            plot=plt.subplot(4,3,plotnumber)
            sns.boxplot(data,x=data[i])
            plt.xlabel(i,fontsize=10)
            plt.ylabel('Priority',fontsize=10)
            plotnumber+=1
plt.tight_layout()
plt.show()


# In[61]:


print(bold+green+"The datasets exhibit outliers,notably in 'No_of_Related_Interactions','No_of_Related_Incidents' and 'No_of_Reassignments'.Employing feature scaling process to mitigate the outliers impact.")


# # Exploratory data analysis(EDA)

# In[62]:


data.describe().T


# In[64]:


features=['CI_Cat','CI_Subcat','Impact','Urgency','Category','Priority']
plt.figure(figsize=(20,20))
plotnumber=1
for feature in features:
    if plotnumber<8:
        plot=plt.subplot(3,2,plotnumber)
        ad=sns.countplot(data=data,x=feature,palette='icefire')
        for i in ad.patches:
            x=i.get_x()+i.get_width()/2
            y=i.get_height()
            ad.annotate(f"{y:.0f}",(x,y),ha='center',fontsize=20)
        plt.xlabel(feature,fontsize=30)
        plt.ylabel("Count",fontsize=30)
        plotnumber+=1
plt.tight_layout()
plt.show()


# In[65]:


print(bold+purple+"All features counts with different records.")


# In[66]:


plt.figure(figsize=(30,40))
plotnumber=1
for i in data:
    if plotnumber<12:
        plot=plt.subplot(4,3,plotnumber)
        sns.histplot(data,x=data[i],kde=True)
        plt.xlabel(i,fontsize=20)
        plt.ylabel("Count",fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[67]:


print(bold+cyan+"All features lies in non-normaly distribution.")


# ### 1. Predicting High Priority Tickets: To predict priority 1 & 2 tickets, so that they can take preventive measures or fix the problem before it surfaces.

# # Feature selection

# In[74]:


corr_mat=data.select_dtypes(include=['number']).corr()
cor=corr_mat['Priority'].sort_values(ascending=True)
print(cor)
# ploting heat map
plt.figure(figsize=(20,20))
sns.heatmap(data=corr_mat,annot=True, cmap='coolwarm',linewidths=1)
plt.show()


# In[69]:


print(bold+green+"Utilizing correlation analysis, heatmap visualization, and domain expertise, carefully choose predictive features for prioritizing ticket resolution based on their impact.")


# # split data

# In[70]:


print(bold+red+'Splitting data in x and y variables.')
X=data.loc[:,['CI_Cat','CI_Subcat','Category','WBS','No_of_Related_Incidents','No_of_Related_Interactions','No_of_Related_Changes']]
y=data.Priority


# In[71]:


print(bold+green+'Data size of X variables'+reset,X.shape)
print(bold+cyan+"Data size of y variables"+reset,y.shape)


# In[72]:


print(bold+purple+"Data split into train and test data.")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)


# In[73]:


print(bold+green+"After splitting of X train data shape:"+reset,X_train.shape)
print(bold+cyan+"After splitting of y train data shape:"+reset,y_train.shape)
print(bold+red+"After splitting of X test data shape:"+reset,X_test.shape)
print(bold+purple+"After splitting of y test data shape:"+reset,y_test.shape)


# # Features Scaling

# In[74]:


X_train_sc=X_train.copy()
X_test_sc=X_test.copy()
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.transform(X_test)
print(bold+green+"Apply Standard scaler techniques on dataset"+reset)


# In[75]:


print(bold+cyan+"Convert numpy array into dataframe.")
X_train_sc_data=pd.DataFrame(X_train_sc)
X_train_sc_data.describe()


# # Balancing dataset

# In[76]:


smote=SMOTE()
X_train_sc_SMOTE,y_train_sc_SMOTE=smote.fit_resample(X_train_sc_data,y_train)
print(bold+green+"Actual y classes: "+reset,Counter(y_train))
print(bold+purple+"Smote standard scaler y classes: "+reset,Counter(y_train_sc_SMOTE))


# # Model Building

# ## Logistic Regression

# In[77]:


# Generate model
log=LogisticRegressionCV(cv=5,max_iter=1000)
# fit training data
log_model_sc=log.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# Predict training data & testing data
y_train_log_data=log_model_sc.predict(X_train_sc_data)
y_test_log_data=log_model_sc.predict(X_test_sc)


# ## Evalution Of Logistic Regression model

# In[78]:


print(classification_report(y_train,y_train_log_data))


# In[79]:


print(classification_report(y_test,y_test_log_data))


# In[80]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_test_log_data)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_test_log_data)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_test_log_data)*100))
print(bold+cyan+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_test_log_data)*100))
print(bold+red+"Testing roc_auc_score:{:.2f}%".format(roc_auc_score(y_test,y_test_log_data)*100))


# In[81]:


pd.crosstab(y_test,y_test_log_data)


# ## Support Vector Classifier

# In[82]:


# build model
svc=SVC()
# train dataset
SVCC=svc.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# Predict training and testing data
y_SVC_training_pred=SVCC.predict(X_train_sc_data)
y_SVC_testing_pred=SVCC.predict(X_test_sc)


# ## Evalution of SVC

# In[83]:


print(classification_report(y_train,y_SVC_training_pred))


# In[84]:


print(classification_report(y_test,y_SVC_testing_pred))


# In[85]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_SVC_testing_pred)*100))
print(bold+cyan+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_SVC_testing_pred)*100))
print(bold+red+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_SVC_testing_pred)*100))
print(bold+purple+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_SVC_testing_pred)*100))
print(bold+red+"Testing roc_auc_score:{:.2f}%".format(roc_auc_score(y_test,y_SVC_testing_pred)*100))


# In[86]:


pd.crosstab(y_test,y_SVC_testing_pred)


# # Decision Tree Clssifier

# In[87]:


# Generate model
dt=DecisionTreeClassifier()
# Fit model with training data
DT=dt.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict training & testing data
y_DT_train_pred=DT.predict(X_train_sc_data)
y_DT_test_pred=DT.predict(X_test_sc)


# ## Evalution of DT

# In[88]:


print(classification_report(y_train,y_DT_train_pred))


# In[89]:


print(classification_report(y_test,y_DT_test_pred))


# In[90]:


print(bold+cyan+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_DT_test_pred)*100))
print(bold+cyan+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_DT_test_pred)*100))
print(bold+green+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_DT_test_pred)*100))
print(bold+red+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_DT_test_pred)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_DT_test_pred)*100))


# In[91]:


pd.crosstab(y_test,y_DT_test_pred)


# ## Hyperparameter tunning of DT

# In[92]:


DT_hyp=DecisionTreeClassifier()
# Hyperparameter tuning of decision tree classifier
param= {"criterion": ('gini','entropy'),
        "max_depth": list(range(2,30)),
        "min_samples_split": list(range(1,6)),
        "min_samples_leaf":[1,2,3,4,5]
        }
DT_cv=GridSearchCV(DT_hyp,param,scoring="accuracy",verbose=1,cv=3)
# DT_clf= Model for training.
# param= hyperparameters (Dictonary created)
# scoring= performance matrix to performance checking.
# verbose= control the verbosity; the more message.
#>1=the computation time for each fold and parameter candidate is displayed;
#>2= the score is also displayed;
#>3= the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
#cv= number of flods

DT_cv.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)   # To training of gridsearch cv.
best_params_per=DT_cv.best_params_    # Give you best parameters.
print(f"Best parameters: {best_params_per}")


# In[136]:


DT_cv_param=DecisionTreeClassifier(criterion="gini",max_depth=29,min_samples_leaf=1,min_samples_split=2)


# In[137]:


# Fit model with training data
DT_param=DT_cv_param.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict data with model
y_DT_train_param=DT_param.predict(X_train_sc_data)
y_DT_test_param=DT_param.predict(X_test_sc)


# ## Evalution of DT

# In[138]:


print(classification_report(y_train,y_DT_train_param))


# In[139]:


print(classification_report(y_test,y_DT_test_param))


# In[140]:


print(bold+cyan+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_DT_test_param)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_DT_test_param)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_DT_test_param)*100))
print(bold+green+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_DT_test_param)*100))
print(bold+cyan+"Testing roc_auc_score:{:.2f}%".format(roc_auc_score(y_test,y_DT_test_param)*100))


# In[141]:


pd.crosstab(y_test,y_DT_test_param)


# # Random Forest

# In[99]:


# Build model
rf=RandomForestClassifier()
# Train dataset
rf_model=rf.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# Predict train & testing data
y_rf_train_pred=rf_model.predict(X_train_sc_data)
y_rf_test_pred=rf_model.predict(X_test_sc)


# ## Evalution of RF

# In[100]:


print(classification_report(y_train,y_rf_train_pred))


# In[101]:


print(classification_report(y_test,y_rf_test_pred))


# In[102]:


print(bold+red+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_rf_test_pred)*100))
print(bold+cyan+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_rf_test_pred)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_rf_test_pred)*100))
print(bold+green+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_rf_test_pred)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_rf_test_pred)*100))


# In[103]:


pd.crosstab(y_test,y_rf_test_pred)


# ## Hyperparametertuning of RF

# In[104]:


# build model
rf_hyp=RandomForestClassifier()

n_estimators= [int(x) for x in np.linspace(start=100,stop=1000,num=15)] # Number of trees used random forest.
max_features= ['sqrt','log2'] # Maximum number of features allowed to try in individual tree
max_depth=[int(x) for x in np.linspace(start=2,stop=90,num=15)]  # Maximum number of depth of iteration
min_sample_split= [2,5,8,10,12]  # Minimum no of sample split 
min_sample_leaf= [1,2,3,4,5,6]  # Minimum number of sample of leaf split
bootstrap= [True,False] # Sampling 

#dictionary for hyperparameters
random_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
             'min_samples_split':min_sample_split,'min_samples_leaf':min_sample_leaf,'bootstrap':bootstrap}

rf_clf_hyp= RandomizedSearchCV(estimator=rf_hyp,scoring='f1',param_distributions=random_grid,n_iter=100,cv=2,
                              verbose=1,n_jobs=-1)
# build h1n1 flue training model
rf_clf_hyp.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# Best parameter 
rf_best_param=rf_clf_hyp.best_params_

print(f'Best parameters: {rf_best_param}')


# In[130]:


rf_param=RandomForestClassifier(n_estimators=807,max_features='sqrt',max_depth=90,min_samples_split=5,min_samples_leaf=1,bootstrap=False)


# In[131]:


# train dataset
rf_hyp_param=rf_param.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict dataset
y_train_hyp_param=rf_hyp_param.predict(X_train_sc_data)
y_test_hyp_param=rf_hyp_param.predict(X_test_sc)


# ## Evalution of RF

# In[132]:


print(classification_report(y_train,y_train_hyp_param))


# In[133]:


print(classification_report(y_test,y_test_hyp_param))


# In[134]:


print(bold+purple+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_test_hyp_param)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_test_hyp_param)*100))
print(bold+cyan+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_test_hyp_param)*100))
print(bold+green+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_test_hyp_param)*100))
print(bold+green+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_test_hyp_param)*100))


# In[135]:


pd.crosstab(y_test,y_test_hyp_param)


# # ANN algorithms

# In[111]:


# Generate model
ANN_model=MLPClassifier(activation='relu',solver="adam")
# fit data with model
ANN=ANN_model.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# Predict datsets
y_ANN_train_pred=ANN.predict(X_train_sc_data)
y_ANN_test_pred=ANN.predict(X_test_sc)


# ## Evalution of ANN algorithms

# In[112]:


print(classification_report(y_train,y_ANN_train_pred))


# In[113]:


print(classification_report(y_test,y_ANN_test_pred))


# In[114]:


print(bold+purple+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_ANN_test_pred)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_ANN_test_pred)*100))
print(bold+red+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_ANN_test_pred)*100))
print(bold+green+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_ANN_test_pred)*100))
print(bold+cyan+"Testing roc_auc_score:{:.2f}%".format(roc_auc_score(y_test,y_ANN_test_pred)*100))


# In[115]:


pd.crosstab(y_test,y_ANN_test_pred)


# ## Hyperparameter tuning of ANN algorithms

# In[116]:


ANN_hyp=MLPClassifier(activation="relu")
param={"hidden_layer_sizes": list(range(100,500,10)),
       "solver": ['lbfgs', 'sgd', 'adam'],
       "alpha":[0.0001,0.001,0.01,1],
       "learning_rate":['constant', 'invscaling', 'adaptive'],
       "learning_rate_init":[0.0001,0.01,1,2],
       "max_iter": list(range(30,1000,10)), 
      }
ANN_cv=RandomizedSearchCV(estimator=ANN_hyp,param_distributions=param,scoring="accuracy",verbose=1,cv=3)

ANN_cv.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)   # To training of gridsearch cv.
best_params_ANN=ANN_cv.best_params_    # Give you best parameters.
print(f"Best parameters: {best_params_ANN}")


# In[119]:


ANN_param=MLPClassifier(solver="lbfgs",max_iter=780,learning_rate_init=1,learning_rate="constant",hidden_layer_sizes=480,
                       alpha=1)


# In[120]:


# fit data with model
ANN_param_hyp=ANN_param.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict datasets
y_ANN_pred_train_hyp=ANN_param_hyp.predict(X_train_sc_data)
y_ANN_pred_test_hyp=ANN_param_hyp.predict(X_test_sc)


# ## Evalution of ANN algorithms

# In[121]:


print(classification_report(y_train,y_ANN_pred_train_hyp))


# In[122]:


print(classification_report(y_test,y_ANN_pred_test_hyp))


# In[123]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_ANN_pred_test_hyp)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_ANN_pred_test_hyp)*100))
print(bold+cyan+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_ANN_pred_test_hyp)*100))
print(bold+red+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_ANN_pred_test_hyp)*100))
print(bold+purple+"Testing roc_auc_score:{:.2f}%".format(roc_auc_score(y_test,y_ANN_pred_test_hyp)*100))


# In[124]:


pd.crosstab(y_test,y_ANN_pred_test_hyp)


# # KNN Algorithms

# In[125]:


# Generate model
knn=KNeighborsClassifier()
# model train with dataset
knn_model=knn.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# Predict dataset
y_knn_pred_train=knn_model.predict(X_train_sc_data)
y_knn_pred_test=knn_model.predict(X_test_sc)


# ## Evalution of KNN

# In[126]:


print(classification_report(y_train,y_knn_pred_train))


# In[127]:


print(classification_report(y_test,y_knn_pred_test))


# In[128]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_knn_pred_test)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_knn_pred_test)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_knn_pred_test)*100))
print(bold+cyan+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_knn_pred_test)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_knn_pred_test)*100))


# In[129]:


pd.crosstab(y_test,y_knn_pred_test)


# # Conclusion
# * All model rest of KNN algorithmns model having false positive rate is low it good sign for prediction of ticket.
# * KNN algorithms model accuracy is 98% and it's false nagative value is 203 that means might be high priority tickets to predict as a low priority tickets so on. 
# * Our best model to predict high priority of ticket is ANN algorithms because 95.24% and false negative value is very low like 51 compare to other model. 

# # 4. Predict RFC (Request for change) and possible failure/misconfiguration of ITSM assets.

# ## feature selection

# In[143]:


corr_rfc=data.select_dtypes(include=['number']).corr()
cor=corr_rfc['No_of_Related_Changes'].sort_values(ascending=True)
print(cor)
# ploting heat map
plt.figure(figsize=(20,20))
sns.heatmap(data=corr_rfc,annot=True, cmap='coolwarm',linewidths=1)
plt.show()


# In[144]:


print(bold+red+"Utilizing correlation analysis, heatmap visualization, and domain expertise, carefully choose predictive features for prioritizing RFC(Request for change) based on their impact.")


# In[150]:


print(bold+purple+'Splitting data in XRFC and yRFC variables.')
XRFC=data.loc[:,['No_of_Reassignments','No_of_Related_Interactions','No_of_Related_Incidents','Priority','CI_Subcat','Category']]
yRFC=data.No_of_Related_Changes


# In[151]:


print(bold+green+'size of XRFC variable: ',XRFC.shape)
print(bold+red+'size of yRFC variable: ',yRFC.shape)


# In[152]:


print(bold+purple+'Split data into train and test dataset')
XRFC_train,XRFC_test,yRFC_train,yRFC_test=train_test_split(XRFC,yRFC,test_size=0.3,random_state=42)


# In[153]:


print(bold+green+"After splitting of XRFC train data shape:"+reset,XRFC_train.shape)
print(bold+cyan+"After splitting of yRFC train data shape:"+reset,yRFC_train.shape)
print(bold+red+"After splitting of XRFC test data shape:"+reset,XRFC_test.shape)
print(bold+purple+"After splitting of yRFC test data shape:"+reset,yRFC_test.shape)


# # Model Building

# ## XGBOOST Algorithms model

# In[157]:


# Model building
xgb= XGBClassifier()
# Train dataset
xgb_model=xgb.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict train and test data
y_xgb_train_pred=xgb_model.predict(X_train_sc_data)
y_xgb_test_pred=xgb_model.predict(X_test_sc)


# ## Evalution of XGBOOST

# In[158]:


print(classification_report(y_train,y_xgb_train_pred))


# In[159]:


print(classification_report(y_test,y_xgb_test_pred))


# In[160]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_xgb_test_pred)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_xgb_test_pred)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_xgb_test_pred)*100))
print(bold+cyan+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_xgb_test_pred)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_xgb_test_pred)*100))


# In[161]:


confusion_matrix(y_test,y_xgb_test_pred)


# ## Hyperparameter tuning of XGBOOST

# In[180]:


# Build model
xgb_hyp=XGBClassifier(random_state=40)
# Hyperparameter
best_param_xgb={'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,0.9],
                'max_depth': [2,3,4,5,7,9,10,12,14,15],
                'max_leaves':[2,3,4,5,6,7,8,9,10],
                'verbosity':[0,1,2,3],
                'gamma':[0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150],
                'n_estimators':[10,20,30,50,65,80,100,115,130,150],
                'reg_alpha': [0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150],
                'reg_lambda': [0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150]
                }
XGB_rcv=RandomizedSearchCV(estimator=xgb_hyp,scoring='accuracy',param_distributions=best_param_xgb,
                   n_iter=100,n_jobs=-1,cv=3,random_state=50,verbose=3)
# Training data of randomizedsearchcv
XGB_rcv.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)

# Best parameters
XGB_best_param=XGB_rcv.best_params_
print(f"best parameters:{XGB_best_param}")


# In[181]:


xgb_bestparam=XGBClassifier(verbosity=2,reg_lambda=0.9,reg_alpha=0.9,n_estimators=65,max_leaves=8,max_depth=10,learning_rate=0.9, gamma=0.3)


# In[182]:


# Train dataset
xgb_model_hyp=xgb_bestparam.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict train and test data
y_xgbhyp_train_pred=xgb_model_hyp.predict(X_train_sc_data)
y_xgbhyp_test_pred=xgb_model_hyp.predict(X_test_sc)


# ## Evalution of XGBOOST hyperparameter

# In[183]:


print(classification_report(y_train,y_xgbhyp_train_pred))


# In[184]:


print(classification_report(y_test,y_xgbhyp_test_pred))


# In[185]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_xgbhyp_test_pred)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_xgbhyp_test_pred)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_xgbhyp_test_pred)*100))
print(bold+cyan+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_xgbhyp_test_pred)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_xgbhyp_test_pred)*100))


# In[186]:


confusion_matrix(y_test,y_xgbhyp_test_pred)


# # Gradient Boosting

# In[175]:


# Build model
grd=GradientBoostingClassifier()
# train datsets
grd_model=grd.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict train and test dataset
y_train_grd_pred=grd_model.predict(X_train_sc_data)
y_test_grd_pred=grd_model.predict(X_test_sc)


# ## Evalution of GradientBooosting

# In[176]:


print(classification_report(y_train,y_train_grd_pred))


# In[177]:


print(classification_report(y_test,y_test_grd_pred))


# In[178]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_test_grd_pred)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_test_grd_pred)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_test_grd_pred)*100))
print(bold+cyan+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_test_grd_pred)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_test_grd_pred)*100))


# In[179]:


confusion_matrix(y_test,y_test_grd_pred)


# ## Hyperparameter tuning of Gradient Boosting algorithms

# In[187]:


grd_hyp=GradientBoostingClassifier(random_state=40)
# Hyperparameter for gridsearch cv
param={'learning_rate':[0.01,0.1,0.5,1],
       'n_estimators':[50,100,150],
       'min_samples_split':[2,3,4],
       'min_samples_leaf':[1,2,3,4],
       'max_depth':[3,4,5,6]
       }
grd_param_hyp=GridSearchCV(estimator=grd_hyp,param_grid=param,verbose=1,scoring='accuracy',cv=2)
# fit data to model
grd_param_hyp.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)

# Best parameter
best_param_hyp=grd_param_hyp.best_params_
print(f"Best parameters: {best_param_hyp}")


# In[188]:


gradientboost_hyp=GradientBoostingClassifier(learning_rate=0.5,max_depth=6,min_samples_leaf=3,min_samples_split=2,n_estimators=150)


# In[189]:


# train datsets
grdhyp_model=gradientboost_hyp.fit(X_train_sc_SMOTE,y_train_sc_SMOTE)
# predict train and test dataset
y_train_grdhyp_pred=grdhyp_model.predict(X_train_sc_data)
y_test_grdhyp_pred=grdhyp_model.predict(X_test_sc)


# ## Evalution of Gradient boosting

# In[191]:


print(classification_report(y_train,y_train_grdhyp_pred))


# In[192]:


print(classification_report(y_test,y_test_grdhyp_pred))


# In[193]:


print(bold+green+"Testing accuracy score: {:.2f}%".format(accuracy_score(y_test,y_test_grdhyp_pred)*100))
print(bold+red+"Testing precision score: {:.2f}%".format(precision_score(y_test,y_test_grdhyp_pred)*100))
print(bold+purple+"Testing recall score:{:.2f}%".format(recall_score(y_test,y_test_grdhyp_pred)*100))
print(bold+cyan+"Testing f1 score:{:.2f}%".format(f1_score(y_test,y_test_grdhyp_pred)*100))
print(bold+red+"Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test,y_test_grdhyp_pred)*100))


# In[194]:


confusion_matrix(y_test,y_test_grdhyp_pred)


# # Conclusion:-
# * Hyper parameter tuning of gradient boosting is high accuracy 94.74%. 
# * fulse negative value is 50 means model predict assets doesn't failure but actually it's failure so it is high risk to predict.
# * Fulse positive value is 685 means model predict failure of assets but actually does not failure. so physically verify it and close this RFC request.

# ## 2. Forecast the incident volume in different fields , quarterly and annual. So that they can be better prepared with resources and technology planning.
# 

# ## Feature selection

# In[63]:


data.info()


# In[62]:


print(bold+green+"As per domain knowledge select features.")
time_series=data.loc[:,['Incident_ID','open_time']]
time_series.head()


# In[63]:


print(bold+purple+"No of incident calculated into count.")
time_series['No_of_Incident']=data.groupby('open_time')['No_of_Related_Incidents'].transform('count')
time_series.head()


# In[64]:


print(bold+red+"Incident ID deleted.")
time_series.drop(['Incident_ID'],axis=1,inplace=True)


# In[65]:


print(bold+red+"Drop duplicate values.")
time_series.drop_duplicates(inplace=True)


# In[66]:


time_series.head()


# In[67]:


print(bold+purple+"Open_time is set as index column.")
time_series=time_series.set_index('open_time')


# In[68]:


time_series.head()


# In[69]:


date_len=len(time_series.index)
print(bold+red+"Total length of date in our dataest:"+reset,date_len)


# In[70]:


print(bold+cyan+"Minimum to maximum date span"+reset)
print(time_series.index.min(),'to',time_series.index.max())


# In[71]:


print(bold+green+"time stamp convert into days"+reset)
data1=time_series['No_of_Incident'].asfreq('D')
data1.head()


# In[72]:


print(bold+green+"Plot graph of time series.")
data1.plot(figsize=(15,6))


# In[73]:


print(bold+red+"Quartly basis sample data."+reset)
quartly=data1.resample('Q').sum()
print(quartly)


# In[74]:


quartly.plot(marker='D',linestyle='-',figsize=(10,6))
plt.title('Quarterly Incident Volume')
plt.xlabel('Quarter')
plt.ylabel('Incident Volume')
plt.show()


# In[75]:


print(bold+purple+"July 2013 to march 2014 have huge incident volume.")


# In[76]:


print(bold+purple+"Year wise sample data."+reset)
year=data1.resample('Y').sum()
print(year)


# In[77]:


year.plot(figsize=(15,6),linestyle='-',marker='D')
plt.title("Yearly Incident Volume")
plt.xlabel("Year")
plt.ylabel("Incident volume")
plt.show()


# In[78]:


print(bold+green+"Maximum ticket incident raised after on 2013-05-06")
max_incident=time_series[time_series.index>dt.datetime(2013,5,6)]
max_incident.head(20)


# In[79]:


def adf_test(series):
    result=adfuller(series)
    print('ADF: {}'.format(result[0]))
    print('pvalue: {}'.format(result[1]))
    print('No of lag: {}'.format(result[2]))
    print('No of observation used for ADF regression and calculate critical values:{}'.format(result[3]))
    print('Critical Value: {}'.format(result[4]))
    for key,value in result[4].items():
        print("\t",key,":",value)
    if result[1]<=0.05:
        print(bold+green+"Strong evidance against null hypothesis,reject null hypothesis and data is stationary."+reset)
    else:
        print(bold+red+"Weak evidence against null hypothesis,accept null hypothesis and data is non-stationary."+reset)
adf_test(max_incident['No_of_Incident'])


# In[80]:


def test_kpss(series):
    resu=kpss(series)
    print('KPSS: {}'.format(resu[0]))
    print('pvalue: {}'.format(resu[1]))
    print('No of lag: {}'.format(resu[2]))
    print('Critical value: {}'.format(resu[3]))
    for key,value in resu[3].items():
        print('\t',key,":",value)
    if resu[1]<=0.05:
        print(bold+green+"Strong evidance against null hypothesis,reject null hypothesis and data is stationary."+reset)
    else:
        print(bold+red+"Weak evidence against null hypothesis,accept null hypothesis and data is non-stationary."+reset)
test_kpss(max_incident['No_of_Incident'])
    


# In[81]:


print(bold+green+"First difference")
max_incident['first difference']=max_incident['No_of_Incident']-max_incident['No_of_Incident'].shift(1)


# In[82]:


max_incident.head(5)


# In[83]:


adf_test(max_incident['first difference'].dropna())


# In[84]:


test_kpss(max_incident['first difference'].dropna())


# In[85]:


acf1=plot_acf(max_incident['first difference'].dropna())
pacf1=plot_pacf(max_incident['first difference'].dropna())


# In[86]:


max_incident.head(15)


# # Split data

# In[87]:


print(bold+green+"time stamp convert into days"+reset)
data2=max_incident['No_of_Incident'].asfreq('D').fillna(0)
data2.head(5)


# In[88]:


# split data into train and test
train_dataset_end=datetime(2013,12,31)
test_dataset_end=datetime(2014,12,31)

train_data=data2[:train_dataset_end]
test_data=data2[train_dataset_end+timedelta(days=1):test_dataset_end]

# predict data
pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]


# In[89]:


print(bold+green+"train data size:",len(train_data))
print(bold+red+"test data size:",len(test_data))
print(bold+purple+"Predict start date"+reset,pred_start_date)
print(bold+cyan+"Predict end date"+reset,pred_end_date)


# # Model selection

# In[108]:


# Build model
AR=ARIMA(train_data,order=(6,1,7))
AR_model=AR.fit()


# In[109]:


print(bold+green+"AIC score of model:",AR_model.aic)


# In[110]:


AR_model.summary()


# In[111]:


forecast_ARI=AR_model.forecast(steps=90)
test1=test_data[:90]
figure,axis=plt.subplots()
plt.plot(forecast_ARI,label='forecast',linestyle='-.',marker='D')
plt.plot(test1,label='test',marker='o')
plt.legend()
figure.autofmt_xdate()


# In[112]:


# Predict model
pred=AR_model.predict(start=pred_start_date,end=pred_end_date)
residual=test_data-pred


# In[113]:


figure,axis=plt.subplots()
plt.plot(pred,label='forecast',linestyle='-.',marker='D')
plt.plot(test1,label='test',marker='o')
plt.legend()
figure.autofmt_xdate()


# In[96]:


print(bold+green+"residual graph have good center tendancy and normal distribution.")
AR_model.resid.plot(kind='kde')


# In[116]:


p=d=q=range(0,9)
pdq=list(itertools.product(p,q,d))
pdq


# In[117]:


list1=[]
list2=[]
for params in pdq:
    try:
        model_arima=ARIMA(train_data,order=params)
        model_fit=model_arima.fit()
        print(params,model_fit.aic)
        list1.append(params)
        list2.append(model_fit.aic)
    except:
        continue


# In[127]:


xy=list2.index(min(list2))
list1[xy]


# In[124]:


# model build
AR_model1=ARIMA(train_data,order=(3,6,3))
# fit with data
ARModel=AR_model1.fit()
forcast_Ar=ARModel.forecast(steps=90)
test2=test_data[:90]
figure,axis=plt.subplots()
plt.plot(forcast_Ar,label='forecast',linestyle='-.',marker='D')
plt.plot(test2,label='test',marker='o')
plt.legend()
figure.autofmt_xdate()


# In[125]:


predict=ARModel.predict(start=pred_start_date,end=pred_end_date)
figure,axis=plt.subplots()
plt.plot(predict,label='forecast',linestyle='-.',marker='D')
plt.plot(test2,label='test',marker='o')
plt.legend()
figure.autofmt_xdate()


# In[126]:


mse=mean_absolute_error(test_data,predict)
print("Mean square error:{:.2f}%".format(mse))
rmse=np.sqrt(mse)
print("root mean square error:{:.2f}%".format(rmse))
mae=mean_squared_error(test_data,predict)
print("Mean square error:{:.2f}%".format(mae))
r2=r2_score(test_data,predict)
print("r2 score:{:.2f}%".format(r2))


# # Seasonal ARIMA include exogenous

# In[146]:


model_sari=SARIMAX(train_data,order=(6,1,7),seasonal_order=(1,1,1,12))
sariw=model_sari.fit()
print(sariw.summary())


# In[147]:


pred_sari=sariw.predict(start=pred_start_date,end=pred_end_date)
pred_sari


# In[148]:


mse=mean_absolute_error(test_data,pred_sari)
print("Mean square error:{:.2f}%".format(mse))
rmse=np.sqrt(mse)
print("root mean square error:{:.2f}%".format(rmse))
mae=mean_squared_error(test_data,pred_sari)
print("Mean square error:{:.2f}%".format(mae))
r2=r2_score(test_data,pred_sari)
print("r2 score:{:.2f}%".format(r2))


# In[149]:


figure,axis=plt.subplots()
plt.plot(pred_sari,label='forecast',linestyle='-.',marker='D')
plt.plot(test_data,label='test',marker='o')
plt.legend()
figure.autofmt_xdate()


# In[135]:


pred_sar=[]
X=train_data.values
for i in test_data:
    sarimax_model=SARIMAX(X,order=(3,6,3))
    sari_model_fit=sarimax_model.fit()
    sari_pred=sari_model_fit.forecast()
    pred_sar.append(sari_pred)
    X=np.append(X,i)    


# In[136]:


mse=mean_absolute_error(test_data,pred_sar)
print("Mean square error:{:.2f}%".format(mse))
rmse=np.sqrt(mse)
print("root mean square error:{:.2f}%".format(rmse))
mae=mean_squared_error(test_data,pred_sar)
print("Mean square error:{:.2f}%".format(mae))
r2=r2_score(test_data,pred_sar)
print("r2 score:{:.2f}%".format(r2))


# In[137]:


figure,axis=plt.subplots()
plt.plot(pred_sar,label='forecast',linestyle='-.',marker='D')
plt.plot(test_data,label='test',marker='o')
plt.legend()
figure.autofmt_xdate()


# # Conclusion:-
# * Seasonal ARIMA include exogenous is good to forecast incident volume.

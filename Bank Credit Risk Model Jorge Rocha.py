
# coding: utf-8

# # Example code for Machine Learning - Credit Risk Model
#   
# This is an extract of code in Python to implement the solution to the exercise for **Risk Assessment**.  
#   
# Main steps included are:
# - Exploration of fields, to discard needed ones and identify categorical / other transformations
# - EDA (Exploratory Data Analysis)
# - Model training and scoring
# - Final proposed solution
#   
# For each step, I will include both in-line comments and mark-up text to explain every decision and action taken.

# ## Load of Train dataset
# Also, we will inspect data and apply transformations and feature removal when apply

# In[128]:


import numpy as np
import pandas as pd

import os
#Models storage
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

#Selección de variables. Usamos sólo este sistema de selección, para no eternizar los pipelines, pero podrían añadirse más
from sklearn.feature_selection import RFECV

#Imports para valoración y entrenamiento de modelos. Clasificación
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Import train dataset, raw, to work with
df_train = pd.read_csv("RAcredit_train.csv", sep=",")

df_train.head(n=40)


# In[3]:


#List of columns of the dataframe
df_columns = df_train.columns
print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# In[4]:


#We'll show general information about types on the dataframe. As we see, there's a number of numeric ones
# and a number of "objects" (strings, or undefined type, likely to act on them to discard / transform)
df_train.info()


# In[5]:


# We'll drop a number of empty columns, as they don't give usable data for the upcoming model. Referred by number
fields_del = [1,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134]
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns[fields_del])]
df_train_aux = df_train.drop(df_columns[fields_del], axis=1)    #dropping of empty / useless columns (1st strike)


# In[6]:


#Refresh of column list and recheck of the status of the dataframe
df_columns = df_train_aux.columns
df_train_aux.head(n=40)


# ## Removal of empty / failed rows
# There is an important number of fields with invalid data. To locate them, we'll lookup NaN values in `loan_amnt` field, as many examples can be seen as follows. Once located, they will be removed prior to go ahead with features checking.

# In[7]:


import math
#With this calculation, we'll check which lines show any problem with load format, causing feature values merging
index_error_rows = [i for i, value in enumerate(df_train_aux.loan_amnt) if math.isnan(value)]
#Result if 1.269 from 77.450 records.

df_train_aux


# In[8]:


aux = df_train_aux.iloc[index_error_rows]
aux


# In[9]:


#We will erase now these rows from df_train_aux
df_train_aux.drop(df_train_aux.index[index_error_rows], inplace=True)


# In[10]:


print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# **Additional fields to be removed**  
# There is a number of fields that are candidates to be removed as well, we'll first show their head and, later on, we'll proceed to act on them:
# - 14, `Issue_d`  ---> *delete*
# - 15, `paymnt_plan` --> *delete, has more than 1.000 wrong values*
# - 16, `url`  --> *delete*
# - 17, `desc`  --> *delete*
# - 18, `purpose`  --> *get different levels and convert to number*
# - 19, `title`  --> *delete* (redundant)
# - 23, `delinq_2yrs`  --> *delete; sparse data, not likely to be used for model*
# - 33, `initial_list_status`  --> *get different levels and convert to number*
# - 34, `out_prncp`  --> *ok*
# - 35, `out_prncp_inv`  --> *ok*
# - 43, `last_pymnt_d`  --> *delete*
# - 44, `last_pymnt_amnt`  --> *ok*
# - 45, `next_pymnt_d`  --> *delete*
# - 49, `policy_code`  --> *delete, most data set to 1.0 with some NaN*
# - 50, `application_type`  --> *get different levels and convert to number*
# - 83, `mths_since_recent_bc`   --> *delete, lot of NaN*
# - 84, `mths_since_recent_bc_dlq`  --> *delete, lot of NaN*
# - 85, `mths_since_recent_inq`  --> *delete, lot of NaN*
# - 86, `mths_since_recent_revol_delinq`    --> *delete, lot of NaN*
#   
# In cases of fields with just some NaN (numeric fields), Imputers could be used to fulfill the gaps. We discard this option in this example, although with more time could be tested.

# In[11]:


#Fields to focus on
fields_del = [14,15,16,17,18,19,23,33,34,35,43,44,45,49,50,83,84,85,86]
df_train_aux[df_columns[fields_del]].head(n=40)


# In[12]:


#Now, we'll see the levels of each value in fields identified, to decide what to do with each one
num_registros = len(df_train_aux) * 1.0   #to convert to float
#Can be used but requires reduction of values ("other" on, all should be "other") and one hot conversion
print("purpose field:")
print(df_train_aux['purpose'].value_counts() / num_registros)
#Can be discarded or converted to 3 states and one hot codification
print("\n\ninitial_list_status field:")
print(df_train_aux['initial_list_status'].value_counts() / num_registros)
#Similar to previous, 3 values and one hot conversion
print("\n\napplication_type field:")
print(df_train_aux['application_type'].value_counts() / num_registros)


# In[13]:


#We transform values of the fields on the previous cell
df_train_aux.purpose[(df_train_aux.purpose != 'debt_consolidation') & (df_train_aux.purpose != 'credit_card') &
                     (df_train_aux.purpose != 'home_improvement')] = 'other'
df_train_aux.initial_list_status[(df_train_aux.initial_list_status != 'w') & 
                                 (df_train_aux.initial_list_status != 'f')] = 'other'
df_train_aux.application_type[(df_train_aux.application_type != 'INDIVIDUAL') &
                              (df_train_aux.application_type != 'JOINT')] = 'OTHER'


# In[14]:


#Now, we drop the columns previously identified
df_train_aux = df_train_aux.drop(df_columns[[16,17,19,23,43,45,49,83,84,85,86]], axis=1)
df_train_aux.head(n=40)


# In[15]:


#we now see some fields that look to be redundant, we'll drop them before going over again to inspect columns
df_train_aux = df_train_aux.drop(labels=['funded_amnt', 'funded_amnt_inv', 'sub_grade'], axis=1)
df_train_aux.head(n=20)


# In[16]:


#We'll recheck again fileds to keep on reviewing fields
df_columns = df_train_aux.columns
print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# In[17]:


#Likely to be dropped, too many categories. With more time, we could check correlation with target, 
# but for this exercise we can ignore it
print("emp_title field:")
print(df_train_aux['emp_title'].value_counts() / num_registros)
#It looks there is an error in certain values of this field, out of the scope of this field. We'll discard field
print("\n\nemp_length field:")
print(df_train_aux['emp_length'].value_counts() / num_registros)
#It looks there is an error in certain values of this field, out of the scope of this field. We'll discard field
print("\n\nhome_ownership field:")
print(df_train_aux['home_ownership'].value_counts() / num_registros)
#It has an error in certain rows, we'll drop it
print("\n\nissue_d field:")
print(df_train_aux['issue_d'].value_counts() / num_registros)


# In[18]:


df_train_aux = df_train_aux.drop(labels=['emp_title', 'emp_length', 'home_ownership','issue_d'], axis=1)


# In[19]:


#We check another group of fieldds now
aux = df_train_aux[['pymnt_plan', 'zip_code', 'addr_state', 'dti', 'earliest_cr_line', 'inq_last_6mths',
                    'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'l_state']]
aux


# In[20]:


#We'll aggregate pymnt_plan values in two groups, as they look to have important relationship with l_state
# In the first one, those with value "n", and the rest with any other value
df_train_aux.pymnt_plan[df_train_aux['pymnt_plan'] == 'n'] = 0
df_train_aux.pymnt_plan[df_train_aux['pymnt_plan'] != 0] = 1


# In[21]:


#We'll drop mostly empty columns, in this case 'mths_since_last_record', 'mths_since_last_delinq', 'addr_state'
#(this one is redundant with zip_code)
df_train_aux = df_train_aux.drop(labels=['addr_state', 'mths_since_last_delinq', 'mths_since_last_record'], axis=1)
df_train_aux


# In[22]:


#Convert zip code to number (3 digits)
list_zip_str = df_train_aux['zip_code']
list_zip_str = [str(zip) for zip in list_zip_str]
#print(list_zip_str[:10])
list_zip_str = [zip[0:3] for zip in list_zip_str]
#print(list_zip_str[:10])
df_train_aux['zip_code_3digit'] = pd.to_numeric(list_zip_str, errors='coerce', downcast='integer')
df_train_aux.drop(labels='zip_code', axis=1, inplace=True)


# In[23]:


df_train_aux['zip_code_3digit']


# In[24]:


#We'll recheck again fileds to keep on reviewing fields
df_columns = df_train_aux.columns
print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# We initiate here another review of a number of fields:
# - `revol_bal`  --> ok
# - `revol_util`  --> ok
# - `total_acc` --> ok
# - `initial_list_status` --> ok, convert to one hot or binary
# - `out_prncp`  --> delete, redundant
# - `out_prncp_inv`  --> ok
# - `total_pymnt`  --> delete, redundant
# - `total_pymnt_inv`  --> ok
# - `total_rec_prncp`  --> ok
# - `total_rec_int`  --> ok
# - `total_rec_late_fee`  --> ok
# - `recoveries`  --> ok
# - `collection_recovery_fee`  --> delete
# - `last_pymnt_amnt`  --> delete
# - `last_credit_pull_d`  --> ok

# In[25]:


fields_del = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
df_train_aux[df_columns[fields_del]]


# In[26]:


#Fields 'collection_recovery_fee' and 'last_pymnt_amnt' are 0, and 'total_pymnt' and 'out_prncp' are
# redundant with their next field, so we remove them
df_train_aux.drop(labels=['collection_recovery_fee','last_pymnt_amnt','total_pymnt','out_prncp'], 
                  axis=1, inplace=True)


# In[27]:


#We'll recheck again fileds to keep on reviewing fields
df_columns = df_train_aux.columns
print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# In[28]:


fields_del = [i for i in range(29, 45)]
fields_del.append(83)
print(fields_del)
df_train_aux[df_columns[fields_del]]


# We initiate here another review of a number of fields:
# - `collections_12_mths_ex_med`  --> delete
# - `mths_since_last_major_derog`  --> delete, many NaN
# - `application_type` --> ok, but needs to be converted to one-hot or numbers
# - `annual_inc_joint` --> delete, many NaN
# - `dti_joint`  --> delete, many NaN
# - `verification_status_joint`  --> delete, many NaN
# - `acc_now_delinq`  --> delete, sparse and many NaN
# - `tot_coll_amt`  --> ok
# - `tot_cur_bal`  --> ok
# - `open_acc_6m`  --> ok
# - `open_il_6m`  --> ok
# - `open_il_12m`  --> ok
# - `open_il_24m`  --> ok
# - `mths_since_rcnt_il`  --> ok
# - `total_bal_il`  --> ok
# - `il_util`  --> ok

# In[29]:


df_train_aux = df_train_aux.drop(labels=['collections_12_mths_ex_med', 'mths_since_last_major_derog',
                                         'annual_inc_joint', 'dti_joint', 'verification_status_joint',
                                         'acc_now_delinq'], axis=1)
#We'll recheck again fileds to keep on reviewing fields
df_columns = df_train_aux.columns
print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# In[30]:


fields_del = [i for i in range(39, 55)]
fields_del.append(77)
print(fields_del)
df_train_aux[df_columns[fields_del]]


# In[31]:


fields_del = [i for i in range(55, 70)]
fields_del.append(77)
print(fields_del)
df_train_aux[df_columns[fields_del]]


# We initiate here another review of a number of fields:
# - `open_rv_12m`  --> ok
# - `open_rv_24m`  --> ok
# - `max_bal_bc` --> ok
# - `all_util` --> ok
# - `total_rev_hi_lim`  --> ok
# - `inq_fi`  --> ok
# - `total_cu_tl`  --> ok
# - `inq_last_12m`  --> ok
# - `acc_open_past_24mths`  --> ok
# - `avg_cur_bal`  --> ok
# - `bc_open_to_buy`  --> ok
# - `bc_util`  --> ok
# - `chargeoff_within_12_mths`  --> delete, sparse
# - `delinq_amnt`  --> delete, sparse
# - `mo_sin_old_il_acct`  --> ok
# - `mo_sin_old_rev_tl_op`  --> ok
# - `mo_sin_rcnt_rev_tl_op`  --> ok
# - `mo_sin_rcnt_tl`  --> ok
# - `mort_acc`  --> ok
# - `num_accts_ever_120_pd`  --> delete, sparse and NaN
# - `num_actv_bc_tl`  --> ok
# - `num_bc_sats`  --> ok
# - `num_bc_tl`  --> delete, redundant with previous field (aprox x2)
# - `num_il_tl`  --> ok
# - `num_op_rev_tl`  --> ok
# - `num_rev_accts`  --> ok
# - `num_rev_tl_bal_gt_0`  --> delete, redundant with previous field
# - `num_sats`  --> ok
# - `num_tl_120dpd_2m`  --> delete, sparse and NaN
# - `num_tl_30dpd`  --> delete, sparse and NaN

# In[32]:


#drop identified fields
df_train_aux = df_train_aux.drop(labels=['chargeoff_within_12_mths', 'delinq_amnt', 'num_accts_ever_120_pd',
                                         'num_bc_tl', 'num_rev_tl_bal_gt_0', 'num_tl_120dpd_2m', 'num_tl_30dpd'],
                                 axis=1)
#We'll recheck again fileds to keep on reviewing fields
df_columns = df_train_aux.columns
print("List of columns:")
_ = [print(str(i) + "\t" + column) for i, column in enumerate(df_columns)]   # _ to avoid printing empty array


# ## NaN search and substitution and Categorical / Date feature process
# Now that we have already explored all features, we'll proceed to transform those ones which need to (categorical data, dates) and to erase or transform NaN values on features. In numerical cases, we'll aply average substitution, for other cases we'll erase the full record

# In[33]:


#'term' field transformation
list_term = list(df_train_aux['term'])
list_term = [(item == ' 36 months') * 1 for item in list_term]   #conversion to categorical, 0 or 1
df_train_aux['cat_term'] = list_term   #add column to DF
df_train_aux.drop(labels=['term'], axis=1, inplace=True)


# In[34]:


#We transform 'grade' column into one-hot set of colunms (7 different values)
aux = pd.get_dummies(df_train_aux['grade'], prefix='grade')
#Addition of one-hot columns
df_train_aux = pd.concat([df_train_aux, aux], axis=1)
#drop 'grade'
df_train_aux.drop(labels=['grade'], axis=1, inplace=True)


# In[35]:


#We transform 'verification status' into categorical (num). 'Verified' and 'Source Verified' --> 1, rest 0
print("verification_status field:")
print(df_train_aux['verification_status'].value_counts() / num_registros)


# In[36]:


df_train_aux.verification_status[(df_train_aux.verification_status == 'Source Verified') | 
                                 (df_train_aux.verification_status == 'Verified')] = 1
df_train_aux.verification_status[(df_train_aux.verification_status != 1)] = 0


# In[37]:


#We transform 'purpose' into categorical (num). One-hot vector conversion
print("purpose field:")
print(df_train_aux['purpose'].value_counts() / num_registros)


# In[38]:


#Conversion to one-hot (4 fields) representation of 'purpose'
aux = pd.get_dummies(df_train_aux['purpose'], prefix='purpose')
#Addition of one-hot columns
df_train_aux = pd.concat([df_train_aux, aux], axis=1)
#We drop the original field ('purpose')
#that was pending (high proportion of NaN)
df_train_aux.drop(labels=['purpose'], axis=1, inplace=True)


# In[39]:


#We transform 'initial_list_status' into categorical (num). W --> 1, rest 0
print("initial_list_status field:")
print(df_train_aux['initial_list_status'].value_counts() / num_registros)


# In[40]:


df_train_aux.initial_list_status[df_train_aux.initial_list_status == 'w'] = 1
df_train_aux.initial_list_status[df_train_aux.initial_list_status != 1] = 0


# In[41]:


#aux function
def convert_date_ecl(date):
    try:
        aux_str = str(date)
        return int(aux_str[4:8])
    except ValueError:
        return 1900

#We transform 'initial_list_status' into categorical (num). W --> 1, rest 0
print("earliest_cr_line field:")
print(df_train_aux['earliest_cr_line'].value_counts() / num_registros)
#we'll convert it to two categories (possible aproach): before 2004 or other data (0) and after or equal 2004 (1)
list_ecl = df_train_aux.earliest_cr_line
list_ecl = [convert_date_ecl(str(row)) for row in list_ecl]
list_ecl = [(row >= 2004) * 1 for row in list_ecl]    #to convert it to 0 or 1, to use as end value
list_ecl
df_train_aux.earliest_cr_line = list_ecl


# In[42]:


#Drop 'recoveries' field, basically 0's
df_train_aux.drop(labels=['recoveries'], axis=1, inplace=True)


# In[43]:


#Auxiliar function
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#We check 'last_credit_pull_d' date field
print("last_credit_pull_d field:")
print(df_train_aux['last_credit_pull_d'].value_counts() / num_registros)
#We'll set as Mar-2017 all numeric values, to simplify
list_lcpd = df_train_aux.last_credit_pull_d
list_lcpd = [isfloat(str(row)) for row in list_lcpd]
df_train_aux.last_credit_pull_d[list_lcpd] = 'Mar-2017'


# In[44]:


#Conversion to one-hot format and substitution on df
aux = pd.get_dummies(df_train_aux['last_credit_pull_d'], prefix='lcpd')
#Addition of one-hot columns
df_train_aux = pd.concat([df_train_aux, aux], axis=1)
#that was pending (high proportion of NaN)
df_train_aux.drop(labels=['last_credit_pull_d'], axis=1, inplace=True)


# In[45]:


#Conversion to categorical of 'application_type' fied
print("application_type field:")
print(df_train_aux['application_type'].value_counts() / num_registros)
#One-hot format
aux = pd.get_dummies(df_train_aux['application_type'], prefix='app_type')
df_train_aux = pd.concat([df_train_aux, aux], axis=1)
df_train_aux.drop(labels=['application_type'], axis=1, inplace=True)


# In[46]:


#Finally, we convert target field to categorical (1 for Default, 0 for Fully Paid)
print("target field:")
print(df_train_aux['l_state'].value_counts() / num_registros)
#Conversion itself. 
df_train_aux.l_state[df_train_aux.l_state == 'Fully Paid'] = 0
df_train_aux.l_state[df_train_aux.l_state == 'Default'] = 1
df_train_aux['target'] = df_train_aux['l_state']  #new field, to be at the end of the dataframe
df_train_aux.drop(labels=['l_state'], axis=1, inplace=True)


# In[47]:


print(df_train_aux['target'].value_counts() / num_registros)


# ### Aggregation in one function of all the steps for feature review, transformation and drop
# This function will serve, once that we have made the necessary work on features, to create a tranformation class for Pipeline and machine learning models

# In[48]:


#All aggregated steps to pass from train or test datasets to the final dataset for modeling work
def transform_dataset_credit_risk(df_in):
    df_out = df_in.copy()   #copy to work with, its result will be returned
    df_columns = df_out.columns
    #1st step
    fields_del = [1,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,
                  125,126,127,128,129,130,131,132,133,134]
    df_out = df_out.drop(df_columns[fields_del], axis=1)   #dropping empty / useless columns (1st strike)
    #2nd step
    df_columns = df_out.columns
    #With this calculation, we'll check which lines show any problem with load format, causing feature values merging
    index_error_rows = [i for i, value in enumerate(df_out.loan_amnt) if math.isnan(value)]
    #Result if 1.269 from 77.450 records.
    #3rd step, delete of rows with invalid data
    df_out.drop(df_out.index[index_error_rows], inplace=True)
    #4th step
    fields_del = [14,15,16,17,18,19,23,33,34,35,43,44,45,49,50,83,84,85,86]
    df_out.purpose[(df_out.purpose != 'debt_consolidation') & (df_out.purpose != 'credit_card') &
                     (df_out.purpose != 'home_improvement')] = 'other'
    df_out.initial_list_status[(df_out.initial_list_status != 'w') & 
                                 (df_out.initial_list_status != 'f')] = 'other'
    df_out.application_type[(df_out.application_type != 'INDIVIDUAL') &
                              (df_out.application_type != 'JOINT')] = 'OTHER'
    df_out = df_out.drop(df_columns[[16,17,19,23,43,45,49,83,84,85,86]], axis=1)
    df_out = df_out.drop(labels=['funded_amnt', 'funded_amnt_inv', 'sub_grade'], axis=1)
    df_out = df_out.drop(labels=['emp_title', 'emp_length', 'home_ownership','issue_d'], axis=1)
    #5th step
    df_columns = df_out.columns
    df_out.pymnt_plan[df_out['pymnt_plan'] == 'n'] = 0
    df_out.pymnt_plan[df_out['pymnt_plan'] != 0] = 1
    df_out = df_out.drop(labels=['addr_state', 'mths_since_last_delinq', 'mths_since_last_record'], axis=1)
    #6th step
    #Convert zip code to number (3 digits)
    list_zip_str = df_out['zip_code']
    list_zip_str = [str(zip) for zip in list_zip_str]
    #print(list_zip_str[:10])
    list_zip_str = [zip[0:3] for zip in list_zip_str]
    #print(list_zip_str[:10])
    df_out['zip_code_3digit'] = pd.to_numeric(list_zip_str, errors='coerce', downcast='integer')
    df_out.drop(labels='zip_code', axis=1, inplace=True)
    df_out.drop(labels=['collection_recovery_fee','last_pymnt_amnt','total_pymnt','out_prncp'], 
                      axis=1, inplace=True)
    df_out = df_out.drop(labels=['collections_12_mths_ex_med', 'mths_since_last_major_derog',
                                 'annual_inc_joint', 'dti_joint', 'verification_status_joint',
                                 'acc_now_delinq'], axis=1)
    df_out = df_out.drop(labels=['chargeoff_within_12_mths', 'delinq_amnt', 'num_accts_ever_120_pd',
                                 'num_bc_tl', 'num_rev_tl_bal_gt_0', 'num_tl_120dpd_2m', 'num_tl_30dpd'],
                         axis=1)
    #7th step
    df_columns = df_out.columns
    #'term' field transformation
    list_term = list(df_out['term'])
    list_term = [(item == ' 36 months') * 1 for item in list_term]   #conversion to categorical, 0 or 1
    df_out['cat_term'] = list_term   #add column to DF
    df_out.drop(labels=['term'], axis=1, inplace=True)
    #We transform 'grade' column into one-hot set of colunms (7 different values)
    aux = pd.get_dummies(df_out['grade'], prefix='grade')
    #Addition of one-hot columns
    df_out = pd.concat([df_out, aux], axis=1)
    #drop 'grade'
    df_out.drop(labels=['grade'], axis=1, inplace=True)
    #8th step
    df_out.verification_status[(df_out.verification_status == 'Source Verified') | 
                               (df_out.verification_status == 'Verified')] = 1
    df_out.verification_status[(df_out.verification_status != 1)] = 0
    #9th step
    #Conversion to one-hot (4 fields) representation of 'purpose'
    aux = pd.get_dummies(df_out['purpose'], prefix='purpose')
    #Addition of one-hot columns
    df_out = pd.concat([df_out, aux], axis=1)
    #We drop the original field ('purpose')
    df_out.drop(labels=['purpose'], axis=1, inplace=True)
    #10th step
    df_out.initial_list_status[df_out.initial_list_status == 'w'] = 1
    df_out.initial_list_status[df_out.initial_list_status != 1] = 0
    #we'll convert it to two categories (possible aproach): before 2004 or other data (0) and after or equal 2004 (1)
    list_ecl = df_out.earliest_cr_line
    list_ecl = [convert_date_ecl(str(row)) for row in list_ecl]
    list_ecl = [(row >= 2004) * 1 for row in list_ecl]    #to convert it to 0 or 1, to use as end value
    df_out.earliest_cr_line = list_ecl
    #11th step
    #last_credit_pull_d transformation (to one-hot)
    list_lcpd = df_out.last_credit_pull_d
    list_lcpd = [isfloat(str(row)) for row in list_lcpd]
    df_out.last_credit_pull_d[list_lcpd] = 'Mar-2017'
    #Conversion to one-hot format and substitution on df
    aux = pd.get_dummies(df_out['last_credit_pull_d'], prefix='lcpd')
    #Addition of one-hot columns
    df_out = pd.concat([df_out, aux], axis=1)
    #We drop the original field ('purpose') and 'mths_since_last_delinq', 'mths_since_last_record'
    #that was pending (high proportion of NaN)
    df_out.drop(labels=['last_credit_pull_d'], axis=1, inplace=True)
    #12th step
    aux = pd.get_dummies(df_out['application_type'], prefix='app_type')
    df_out = pd.concat([df_out, aux], axis=1)
    df_out.drop(labels=['application_type'], axis=1, inplace=True)
    #Finally, we convert target field to categorical (0 for Default, 1 for Fully Paid)
    df_out.l_state[df_out.l_state == 'Fully Paid'] = 1
    df_out.l_state[df_out.l_state == 'Default'] = 0
    df_out['target'] = df_out['l_state']  #new field, to be at the end of the dataframe
    df_out.drop(labels=['l_state'], axis=1, inplace=True)
    
    return df_out    #result of the process


# In[49]:


#We regenerate the df for training, processed by the function
df_train = pd.read_csv("RAcredit_train.csv", sep=",")
df_train_aux = transform_dataset_credit_risk(df_train)


# ### Correlation matrix
# Now that we have processed df, we can take a look to correlation matrix between all features:

# In[50]:


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

corr_matrix = df_train_aux[df_train_aux.columns[0:15]].corr()

# Figura dimensions
fig = plt.figure(1, figsize=(10,10))
# Only one subplot (111)
ax = fig.add_subplot(111)
labels = ['labels']+corr_matrix.columns.tolist()
ax.set_xticklabels(labels, rotation='45', ha='left')
ax.set_yticklabels(labels, rotation='horizontal', ha='right')

corr_mat_plot = ax.matshow(corr_matrix, cmap=plt.cm.hot_r)
# Con esto indicamos explicitamente que el rango de nuestros valores será -1,1
corr_mat_plot.set_clim([-1,1])
cb = fig.colorbar(corr_mat_plot)
cb.set_label("Correlation Coefficient")

plt.show()


# With an analysis like this, we can detect those features that are likely to be removed because, although they are valid in terms of format and completeness, are redundant with other features. Here is shown an example of this: **`loan_amnt` and `installment` have almost 1 correlation, so one of them can be dropped.**  
# This process, having up to 90 features once processed, it's better suited analyzing corr matrix, although less graphical

# In[51]:


df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[0:71]].copy()
corr_matrix = aux.corr()   #full correlation matrix
corr_matrix


# In[52]:


df_columns = aux.columns   #to not considerate one-hot columns
cm = corr_matrix.as_matrix()   #to have a bidimensional matriz, instead of DataFrame
print(cm.shape)

#We'll identify those columns (outside diagonal, self) that present over 0.9 correlation, as candidates to be dropped
def check_correls(df_columns, cm, size):
    list_columns_corr = []
    for r, row in enumerate(df_columns):
        if r >= size:   #ignore one-hot fields
            continue
        for c, col in enumerate(df_columns):
            if c != r:
                if r > c:  #to get only lower diagonal of the matrix
                    corr = abs(cm[r,c])
                    if corr > 0.90:
                        list_columns_corr.append((row, col))
    return list_columns_corr
check_correls(df_columns, cm, cm.shape[0])    #use of a function to repeat it once df is revisited


# With this data, we can conclude that a second feature reduction should be reasonable; with **over 0.9 correlation** we will loose little or no information for the models if we drop one of the fields. We'll drop the following ones based on this:
# - `installment`
# - `inq_last_6mths`
# - `pub_rec`
# - `total_rec_int`
# - `open_il_6m`
# - `total_bal_il`
# - `mort_acc`
# - `num_sats`
# - `pct_tl_nvr_dlq`
#   
# A new version of the processing function is created, to drop all these fields

# In[53]:


def transform_dataset_credit_risk_rev(df_in):
    df_out = transform_dataset_credit_risk(df_in)  #to start from the previous point
    df_out.drop(labels=['installment', 'inq_last_6mths', 'pub_rec', 'total_rec_int',
                        'open_il_6m', 'total_bal_il', 'mort_acc', 'num_sats', 'pct_tl_nvr_dlq'],
                axis=1, inplace=True)
    return df_out

df_train_aux = transform_dataset_credit_risk_rev(df_train)   #necessary to start over

df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[0:62]].copy()
corr_matrix = aux.corr()   #full correlation matrix
corr_matrix


# In[54]:


df_columns = aux.columns   #to not considerate one-hot columns
cm = corr_matrix.as_matrix()   #to have a bidimensional matriz, instead of DataFrame
print(cm.shape)

check_correls(df_columns, cm, cm.shape[0])


# ## NaN Check
# Check about `NaN` values in Dataset

# In[55]:


#With this method, we get a matrix in which every cell answers False (not NaN) or True (NaN)
nan_matrix = df_train_aux.isnull()
print(nan_matrix.shape)

for i, column in enumerate(nan_matrix.columns):
    col = nan_matrix[column]
    print(str(column) + ", tiene " + str(sum(col)) + " nulos")


# Several features present NaN values; for those with more than 1.000 (just to set a value) we'll drop the feature. For those with NaN values under that threshold, we'll impute *average* value of the feature. In a real, production intended model, we should first try to get more information about each feature to impute, because in certain cases a *median* impute strategy could be the best option (most common value), of *average*, or *0* or even just dropping the full row.

# In[56]:


#Still some features to be removed, same process
#First line are correlation fields, the second one are fields that have too many NaN (over 1.000)
def transform_dataset_credit_risk_rev2(df_in):
    df_out = transform_dataset_credit_risk_rev(df_in)  #to start from the previous point
    df_out.drop(labels=['id', 'dti', 'open_acc', 'mths_since_rcnt_il', 'open_rv_12m', 'num_op_rev_tl'],
                axis=1, inplace=True)
    df_out.drop(labels=['il_util', 'mo_sin_old_il_acct'], axis=1, inplace=True)
    #Features to be removed, once EDA analysis (graphical)
    df_out.drop(labels=['open_il_12m', 'open_acc_6m', 'open_rv_24m', 'inq_fi', 'total_cu_tl', 'inq_last_12m'], 
                axis=1, inplace=True)
    return df_out

df_train_aux = transform_dataset_credit_risk_rev2(df_train)   #start over


# In[57]:


#Recalculate NaN status, and impute strategy. We focus only in fields with NaN (avoid rest)
nan_matrix = df_train_aux.isnull()
print(nan_matrix.shape)

#list_NaN_fields = []
for i, column in enumerate(nan_matrix.columns):
    col = nan_matrix[column]
    if sum(col) > 0:
        print(str(column) + ", tiene " + str(sum(col)) + " nulos")

imputer = Imputer() #by default, uses mean
imputer.fit(df_train_aux)
df_imputer = imputer.transform(df_train_aux)

#Rebuild df_train_aux, now withour NaN
df_train_aux = pd.DataFrame(df_imputer, columns=nan_matrix.columns)
print("\n\n----------------------------------------------")
print("Recheck of NaN features:")
nan_matrix = df_train_aux.isnull()
for i, column in enumerate(nan_matrix.columns):
    col = nan_matrix[column]
    if sum(col) > 0:
        print(str(column) + ", tiene " + str(sum(col)) + " nulos")
print("-------------------------------------------")
print("End of recheck of NaN")


# In[58]:


df_columns = df_train_aux.columns
fields_del = [i for i in range(20, 40)]
print(fields_del)
df_train_aux[df_columns[fields_del]]


# In[59]:


df_train_aux


# ## EDA
# Once we have treated features, analyzed correlation for further feature discard, and treated NaN values as well, we can now take a look to the distribution of features.  
# In a real scenario, we should review carefully all features (once made this first debranching of features), as valuable information could be found: outlier information / concentration on borders, non-logical distributions, gaps on data bins, etc. Here is an example:

# In[60]:


#Let's split df into pieces for easier EDA on each one:
df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[0:16]]
with plt.xkcd():
    aux.hist(bins=30, figsize=(20,15))
    plt.show()


# In[61]:


df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[16:32]]
with plt.xkcd():
    aux.hist(bins=30, figsize=(20,15))
    plt.show()


# In[62]:


df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[32:48]]
with plt.xkcd():
    aux.hist(bins=30, figsize=(20,15))
    plt.show()


# Many fields, just like `all_util`, look to have outliers, maybe that due to errors in the structure of data in same rows (too many field separators). Let's try to identify them:

# In[63]:


aux_mean_threshold = abs(df_train_aux.all_util.mean() * 3)   #we consider 3 times the mean as erroneous / outlier value
list_erroneous_rows = list(df_train_aux['all_util'])
list_erroneous_rows = [abs(value) > aux_mean_threshold for value in list_erroneous_rows]
list_erroneous_rows = [i for i, value in enumerate(list_erroneous_rows) if value]   #to get indexes
df_train_aux['all_util'].describe()
df_train_aux = df_train_aux.drop(list_erroneous_rows, axis=0)

df_columns = df_train_aux.columns
fields_del = [i for i in range(20, 40)]
print(fields_del)
df_train_aux[df_columns[fields_del]]


# ### Feature scale transformation
# Before going on with sklearn Transformation class (to inject on a Pipeline) we'll drop a number o columns that are very influenced by extreme outliers (no information enough to transform or treat them) and some others that would befave better with a *logatirmic* or *sqrt*. Al feature graphics of this variables don't seem to follow a Normal or Exponential distribution, they should improve when transformed.
# The list of features is as follows:
# - open_il_12m    *drop*
# - open_acc_6m    *drop*
# - open_rv_24m    *drop*
# - inq_fi         *drop*
# - total_cu_tl    *drop*
# - inq_last_12m   *drop*
# - verification_status
# - earliest_cr_line
# - revol_bal
# - revol_util
# - out_prncp_inv
# - total_pymnt_inv
# - total_rec_prncp
# - total_rec_late_fee
# - recoveries
# - tot_coll_amt
# - open_il_24m
# - max_bal_bc
# - total_rev_hi_lim
# - avg_cur_bal
# - bc_open_to_buy
# - bc_util
# - mo_sin_rcnt_tl
# - num_tl_90g_dpd_24m
# - pub_rec_bankruptcies
# - tot_hi_cred_lim
# - total_bal_ex_mort
# - total_bc_limit
# - total_il_high_credit_limit
#   
# Drop fields are included in the previous transformation function, to make them permanent. In the following cell, we'll make a test on transforming these fields.  
#   
# **Further work with features (just like percent_bc_gt_75, among others)

# In[64]:


#List of fields to be transformed (sqrt)
list_fields_sqrt = ['verification_status', 'earliest_cr_line', 'revol_bal', 'revol_util', 'out_prncp_inv',
                    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_late_fee', 'recoveries',
                    'tot_coll_amt', 'open_il_24m', 'max_bal_bc', 'total_rev_hi_lim',
                    'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_rcnt_tl', 'num_tl_90g_dpd_24m',
                    'pub_rec_bankruptcies', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 
                    'total_il_high_credit_limit']
#For each of the columns on the list, whe apply np.sqrt() transformation
for col in list_fields_sqrt:
    df_train_aux[col] = df_train_aux[col].map(np.sqrt)


# In[65]:


#Let's try to transform one of the variables that is showing big outliers / variation (annual_inc)
#with a Logaritmic transformation. First, let's see its main statistical parameters
for col in df_train_aux.columns:
    print(col)
    print(df_train_aux[col].describe())

#df_train_aux['annual_inc'] = df_train_aux['annual_inc'].map(np.sqrt)


# In[66]:


#Let's split df into pieces for easier EDA on each one:
df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[0:16]]
with plt.xkcd():
    aux.hist(bins=30, figsize=(20,15))
    plt.show()


# In[67]:


df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[16:32]]
with plt.xkcd():
    aux.hist(bins=30, figsize=(20,15))
    plt.show()


# In[68]:


df_columns = df_train_aux.columns
aux = df_train_aux[df_columns[32:48]]
with plt.xkcd():
    aux.hist(bins=30, figsize=(20,15))
    plt.show()


# # Model for Credit Risk Evaluation
# Once we have reviewd features, done EDA and made our tests with feature transformations (SQRT), it's time to build our models and to check them against *test* DataSet to evaluate. There are some considerations to have in mind at this point:
# - ***Transformations*** --> In order to simplify the process of treatment both for *train* and *test*, as well as applying sqrt feature scaling. After that, an *Imputer* is used as previously to fulfull NaN gaps over the dataset. Also, we'll use a *StandardScaler* from sklearn in order to facilitate linear regression work over features. This will be 2 steps on Pipeline
# - ***Model/s*** --> once transformed, we'll apply at least *LinearRegression* classifier.
# - ***GridSearchCV*** --> for model selection, we'll make use of cross validation approach, with hiperparameters, to pick the best parameter configuration on tested mdoels.
# - ***Test dataset evaluation*** --> as the last step, we'll evaluate behaviour against test dataset. For model evaluation we must have in mind that it is a binary classification problem, with very unbalanced class weight (> 96% paid credits, 1, and < 4% default 0), so there are two main posibilities; to strictly focus on *Recall* (ability of the model to not miss on *1* prediction, or F1-score, which gives the same relevance to accuracy on 0's and 1's prediction.

# In[74]:


#Class for data transformation of the dataset.
from sklearn.base import BaseEstimator, TransformerMixin
class Transform_DS_CreditRisk(TransformerMixin, BaseEstimator):
    def __init__(self, list_sqrt_tf_fields = []):
        self.list_sqrt_tf_fields = list_sqrt_tf_fields
    # Implements fit_transform(), fit() and transform()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # To transform, we apply the function transform_dataset_credit_risk_rev2, later on the Imputer
        aux = transform_dataset_credit_risk_rev2(X)

        nan_matrix = aux.isnull()
        imputer = Imputer() #by default, uses mean
        imputer.fit(aux)
        df_imputer = imputer.transform(aux)

        #Rebuild df_train_aux, now withour NaN
        df_out = pd.DataFrame(df_imputer, columns=nan_matrix.columns)

        #For each of the columns on the list, whe apply np.sqrt() transformation
        for col in self.list_sqrt_tf_fields:
            df_out[col] = df_out[col].map(np.sqrt)
        
        return df_out


# Pipeline and Model creation. To start, only LinearRegression(), with *recall* scoring, to avoid at maximum to mark as *Fully paid* cases that are really *Default*  
#   
# Also, we have to consider that `RAcredit_test.csv` file has the same structure as the training one, but target field is empty, so it can only can be used to predict with the model, just like a production scenario. Thus, for validation we'll need to split training file in 2 sets. We'll separate an 80% for cross validation training and 20% for test validation.

# In[131]:


#List of models and results for final comparison
lista_modelos_clasificacion = []
#Also, we use the Transformer to get Y_train, Y_test at least
df_in = pd.read_csv("RAcredit_train.csv", sep=",")
df_target = pd.read_csv("RAcredit_test.csv", sep=",")
CRT = Transform_DS_CreditRisk()   #without sqrt feature transformation
CRT_sqrt = Transform_DS_CreditRisk(list_fields_sqrt)   #with sqrt feature transformation
df_train, df_test = train_test_split(df_in, train_size=0.8)
#Aux transformation for train and test
df_train_aux = CRT.transform(df_train)
df_test_aux = CRT.transform(df_test)
df_train_aux_sqrt = CRT_sqrt.transform(df_train)
df_test_aux_sqrt = CRT_sqrt.transform(df_test)
#We prepare y for train (fit)
y_train = df_train_aux['target']
y_test = df_test_aux['target']


# In[122]:


print(y_train.shape)
print(df_train_aux.shape)
print(sum(y_train))


# In[115]:


#df_train[df_train.columns[0:-1]]


# In[ ]:


#First, we create transforming objects: Transform_DS_CreditRisk and StandardScaler
CR_transformer = Transform_DS_CreditRisk()   #to apply sqrt on every identified field
scaler = StandardScaler()
rfecv = RFECV(estimator=DecisionTreeRegressor())   #to select variables
#We create Logistic Regressor for Clasification
logreg_classif = LogisticRegression()
#We now create the Pipeline
pipeline_logreg = Pipeline(steps=[('scaler', scaler), ('RFECV', rfecv), ('logregcl', logreg_classif)])  
#('CRTf', CR_transformer), 

#Hiperparameters to feed the GridSearchCV Pipepline
hyperparams_pipeline = {
#    "CRTf__list_sqrt_tf_fields": [list_fields_sqrt, None],
    "RFECV__cv": [3],
    "RFECV__scoring": ["neg_mean_squared_error"],
    "scaler__with_mean":[True, False],
    "scaler__with_std":[True, False],
    "logregcl__C": [0.00001, 0.001, 1],
    "logregcl__dual": [False],
    "logregcl__class_weight": ["balanced"],
    "logregcl__fit_intercept": [True]
}

#Creation of GridSearchCV with all parameters
logreg_gs = GridSearchCV(pipeline_logreg, param_grid=hyperparams_pipeline, scoring="recall",
                         cv=5, verbose=1, n_jobs=-1)
#Model fit with df_train
logreg_gs.fit(X=df_train_aux[df_train_aux.columns[0:-1]],
              y=y_train)


# In[135]:


####


# In[165]:


lista_modelos_clasificacion = []
#Trained. Let's print best scoring and model parameters
print("Best scoring:")
print(logreg_gs.best_score_)
print("Best parameters:")
print(logreg_gs.best_estimator_)
print("RFECV selected the following features as segnificant (bool list):")
featuresRFECV_boolean = logreg_gs.best_estimator_.named_steps['RFECV'].support_
print(featuresRFECV_boolean)
list_selected_features = df_train_aux.columns[featuresRFECV_boolean]
print(list_selected_features)
print("Best params of LogisticRegression:")
print(logreg_gs.best_estimator_.named_steps['logregcl'].coef_)
print("General data for LogisticRegression:")
print(logreg_gs.best_estimator_.named_steps['logregcl'])

#Insertamos en la lista de modelos probados
lista_modelos_clasificacion.append(("Logistic Regression", logreg_gs.best_score_, logreg_gs.best_estimator_))


# In[166]:


#Once that best hyperparameters are identified, we'll reduce options, in order to boost up solution.
#In this case, we potenciate accuracy on 0's (Default on this example), instead of 1's

#Hiperparameters to feed the GridSearchCV Pipepline
hyperparams_pipeline = {
#    "CRTf__list_sqrt_tf_fields": [list_fields_sqrt, None],
    "RFECV__cv": [3],
    "RFECV__scoring": ["neg_mean_squared_error"],
    "scaler__with_mean":[True, False],
    "scaler__with_std":[True],
    "logregcl__C": [0.00001],    #less params to improve performance
    "logregcl__dual": [False],
    "logregcl__class_weight": ["balanced"],
    "logregcl__fit_intercept": [True]
}

#Creation of GridSearchCV with all parameters
#Scoring is now "precision", that focuses on not to fail when class is 0
logreg_gs_precision = GridSearchCV(pipeline_logreg, param_grid=hyperparams_pipeline, scoring="precision",
                                   cv=5, verbose=1, n_jobs=-1)
#Model fit with df_train
logreg_gs_precision.fit(X=df_train_aux[df_train_aux.columns[0:-1]],
                        y=y_train)


# In[167]:


#Trained. Let's print best scoring and model parameters
print("Best scoring:")
print(logreg_gs_precision.best_score_)
print("Best parameters:")
print(logreg_gs_precision.best_estimator_)
print("RFECV selected the following features as segnificant (bool list):")
featuresRFECV_boolean = logreg_gs_precision.best_estimator_.named_steps['RFECV'].support_
print(featuresRFECV_boolean)
list_selected_features = df_train_aux.columns[featuresRFECV_boolean]
print(list_selected_features)
print("Best params of LogisticRegression:")
print(logreg_gs_precision.best_estimator_.named_steps['logregcl'].coef_)
print("General data for LogisticRegression:")
print(logreg_gs_precision.best_estimator_.named_steps['logregcl'])

#Insertamos en la lista de modelos probados
lista_modelos_clasificacion.append(("Logistic Regression - precision", logreg_gs_precision.best_score_, 
                                    logreg_gs_precision.best_estimator_))


# In[169]:


#Once that best hyperparameters are identified, we'll reduce options, in order to boost up solution.
#In this case, we use "f1" scoring (balanced between )

#Hiperparameters to feed the GridSearchCV Pipepline
hyperparams_pipeline = {
#    "CRTf__list_sqrt_tf_fields": [list_fields_sqrt, None],
    "RFECV__cv": [3],
    "RFECV__scoring": ["neg_mean_squared_error"],
    "scaler__with_mean":[True, False],
    "scaler__with_std":[True],
    "logregcl__C": [0.00001],    #less params to improve performance
    "logregcl__dual": [False],
    "logregcl__class_weight": ["balanced"],
    "logregcl__fit_intercept": [True]
}

#Creation of GridSearchCV with all parameters
#Scoring is now "f1", that combines recall and precision
logreg_gs_f1 = GridSearchCV(pipeline_logreg, param_grid=hyperparams_pipeline, scoring="f1",
                            cv=5, verbose=1, n_jobs=-1)
#Model fit with df_train
logreg_gs_f1.fit(X=df_train_aux[df_train_aux.columns[0:-1]],
                 y=y_train)


# In[170]:


#Trained. Let's print best scoring and model parameters
print("Best scoring:")
print(logreg_gs_f1.best_score_)
print("Best parameters:")
print(logreg_gs_f1.best_estimator_)
print("RFECV selected the following features as segnificant (bool list):")
featuresRFECV_boolean = logreg_gs_f1.best_estimator_.named_steps['RFECV'].support_
print(featuresRFECV_boolean)
list_selected_features = df_train_aux.columns[featuresRFECV_boolean]
print(list_selected_features)
print("Best params of LogisticRegression:")
print(logreg_gs_f1.best_estimator_.named_steps['logregcl'].coef_)
print("General data for LogisticRegression:")
print(logreg_gs_f1.best_estimator_.named_steps['logregcl'])

#Insertamos en la lista de modelos probados
lista_modelos_clasificacion.append(("Logistic Regression - F1 scoring", logreg_gs_f1.best_score_, 
                                    logreg_gs_f1.best_estimator_))


# In[ ]:


#######################################################################################################
# PENDING, DUE TO LONGLASTING EXECUTION
#######################################################################################################
#Now, same parameters, but with sqrt applied to features
#Creation of GridSearchCV with all parameters
#logreg_gs_sqrt = GridSearchCV(pipeline_logreg, param_grid=hyperparams_pipeline, scoring="recall",
#                         cv=10, verbose=1, n_jobs=-1)
#Model fit with df_train
#logreg_gs_sqrt.fit(X=df_train_aux_sqrt[df_train_aux_sqrt.columns[0:-1]],
#              y=y_train)

#Trained. Let's print best scoring and model parameters
#print("Best scoring:")
#print(logreg_gs.best_score_)
#print("Best parameters:")
#print(logreg_gs.best_estimator_)

#Insertamos en la lista de modelos probados
#lista_modelos_clasificacion.append(("Logistic Regression_sqrt", logreg_gs.best_score_, 
#                                    logreg_gs.best_estimator_))


# ### Result analysis
# Once that we have calculated and evaluated models with train sets, we can launch `predict` method and check results

# In[171]:


#We'll get each model and compare predictions in test dataset
#1st --> model without sqrt feature transformation
for model in lista_modelos_clasificacion:
    #model = lista_modelos_clasificacion[0]
    model_name = model[0]
    model_score = model[1]
    model_be = model[2]
    print("\nModel: " + model_name)
    y_predict = model_be.predict(df_test_aux[df_test_aux.columns[0:-1]])
    list_diff_predictions = y_predict == y_test    #list of booleans
    print("Total elements: " + str(len(y_test)) + ", with " +  str(sum(list_diff_predictions)) + " successfull")
    print("% of total correct results: " + str(sum(list_diff_predictions) / len(y_test)))
    #Now, we filter the analysis for 'Default' records
    y_test_index = [y == 1.0 for y in list(y_test)]

    list_1_test = [y == 1.0 for y in y_test]
    list_0_test = [y == 0.0 for y in y_test]
    print("Total coincidences for 1: " + str(sum(y_predict[list_1_test])) + " from " + str(sum(list_1_test)))
    print("\t" + str(sum(y_predict[list_1_test]) / sum(list_1_test)) + "%")
    print("Total coincidences for 0: " + str(sum(y_predict[list_0_test])) + " from " + str(sum(list_0_test)))
    print("\t" + str(sum(y_predict[list_0_test]) / sum(list_0_test)) + "%")


# ### Execution of the Model (F1 or Recall) over Test Dataset
# Once that we have made our tests (we should try other models as well, but I have no time to launch them), it's time to apply it to the test dataset, without known result

# In[186]:


#Get the dataset transformed to pass it to predict
df_target['l_state'] = 'Default'    #transform function expects 'Default' or 'Fully Paid', and this DS is all NaN
df_target_aux = CRT.transform(df_target)

for model_item in lista_modelos_clasificacion:
    model_name = model_item[0]
    model_score = model_item[1]
    model_be = model_item[2]
    y_target = model_be.predict(df_target_aux[df_target_aux.columns[0:-1]])
    #Convert to boolean array
    y_target = [y == 1.0 for y in y_target]
    y_target_default_indexes = []    #list to store indexes, one for each
    y_target_paid_indexes = []
    for i, y in enumerate(y_target):
        if y:
            y_target_paid_indexes.append(i)
        else:
            y_target_default_indexes.append(i)
    print("\n-----------------------------------------------------------------------------")
    print("MODEL " + model_name)
    print("-----------------------------------------------------------------------------\n")
    print("PREDICTIONS:")
    print("Fully paid: " + str(len(y_target_paid_indexes)) + " of " + str(len(y_target)) + ", " + 
          str(len(y_target_paid_indexes) / len(y_target)) + "% of total")
    print("Default: " + str(len(y_target_default_indexes)) + " of " + str(len(y_target)) + ", " + 
          str(len(y_target_default_indexes) / len(y_target)) + "% of total")


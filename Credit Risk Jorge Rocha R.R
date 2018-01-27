########################################################################################################################
## Exercise for anlyzing data and preparing a Model of risk management in credits, based on the file RAcredit_train.csv
## By: Jorge Rocha Blanco
########################################################################################################################

#Working directory
setwd("~/Python Notebooks/Smava")

#Lybraries
library(sqldf)
library(lubridate)
library(norm)
library(corrplot)
library(reshape2)
library(ggplot2)
library(lmtest)

# We load first of all the file into a dataframe, to ease our work on it
df_train <- read.csv(file = 'RAcredit_train.csv', header = TRUE, sep = ',', stringsAsFactors = FALSE)
head(df_train)
#As shown in RStudio, the dataset has 77450 rows with 136 features. There are a number of cases to work with prior to 
#be able to build a linear regression or a machine learning model.
#-Features with important % of NA data --> to be dropped
#-Features that require transformation: cathegorical labels (with or without erronous values within), date, non-valid 
#numerical features (e.g. ZIP code with 'xxx'), features that the same or a simple transformation from other (visually),
#-Features that require NA values correction
#-Rows that present particular problem (e.g. rows with high proportion of NA values that need to be dropped)

## We'll go on this data preparation now, to have the data ready for model / regression approach
colnames(df_train)       #Show all colnames of the dataframe

#1st step: we drop a number of columns that are useless due to emptyness. We create df_train_aux 
df_train_aux <- df_train[, -c(2,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135)]
colnames(df_train_aux)
#2nd step: row with invalid data removal. Through filtering NA data in the column "loan_amnt"
df_train_aux <- subset(df_train_aux, !is.na(df_train_aux$loan_amnt))
#3rd step: additional features have to be deleted (NA, high frequency of erroneous data, sparse / useless features, redundant) 
# and check on other fields to see their different (unique) values
fields_to_process <- c(15,16,17,18,19,20,24,34,35,26,44,45,46,50,51,84,85,86,87)
aux <- df_train_aux[fields_to_process]
#Unique values checks
#We will check unique values of fields
sqldf("SELECT purpose as p, count(purpose) as tot FROM df_train_aux GROUP BY purpose ORDER BY tot DESC")
sqldf("SELECT initial_list_status, count(initial_list_status) as tot FROM df_train_aux GROUP BY initial_list_status ORDER BY tot DESC")
sqldf("SELECT application_type, count(application_type) as tot FROM df_train_aux GROUP BY application_type ORDER BY tot DESC")
#We must transform these three fields in order to use them
## --> purpose
df_train_aux$purpose[df_train_aux$purpose != "debt_consolidation" & df_train_aux$purpose != "credit_card" & 
                       df_train_aux$purpose != "home_improvement"] = "other"
df_train_aux$fact_purpose <- as.numeric(factor(df_train_aux$purpose))   #convert into numeric (one per category)
## --> initial_list_status
df_train_aux$initial_list_status[df_train_aux$initial_list_status != "w" & df_train_aux$initial_list_status != "f"] = "other"
df_train_aux$fact_initial_list_status <- as.numeric(factor(df_train_aux$initial_list_status))
## --> application_type
df_train_aux$application_type[df_train_aux$application_type != "INDIVIDUAL" & df_train_aux$application_type != "JOINT"] = "OTHER"
df_train_aux$fact_application_type <- as.numeric(factor(df_train_aux$application_type))
#Columns drop
fields_to_process <- c(17,18,20,24,44,46,50,84,85,86,87)
df_train_aux <- df_train_aux[,-fields_to_process]
df_train_aux$purpose <- NULL
df_train_aux$initial_list_status <- NULL
df_train_aux$application_type <- NULL
## 4th step
colnames(df_train_aux)
df_train_aux$funded_amnt <- NULL
df_train_aux$funded_amnt_inv <- NULL
df_train_aux$sub_grade <- NULL
## 5th step
# We have to check different values for several fields
sqldf("SELECT emp_title, count(emp_title) as tot FROM df_train_aux GROUP BY emp_title HAVING tot > 559 ORDER BY tot DESC")
sqldf("SELECT emp_length, count(emp_length) as tot FROM df_train_aux GROUP BY emp_length HAVING tot > 5 ORDER BY tot DESC")
sqldf("SELECT home_ownership, count(home_ownership) as tot FROM df_train_aux GROUP BY home_ownership HAVING tot > 1 ORDER BY tot DESC")
sqldf("SELECT issue_d, count(issue_d) as tot FROM df_train_aux GROUP BY issue_d HAVING tot > 1 ORDER BY tot DESC")
df_train_aux$emp_title <- NULL    #4 features to be dropped (very disperse categories)
df_train_aux$emp_length <- NULL
df_train_aux$home_ownership <- NULL
df_train_aux$issue_d <- NULL
## 6th step
#We are going to transform and aggregate Payment plan field
sqldf("SELECT pymnt_plan, count(pymnt_plan) as tot FROM df_train_aux GROUP BY pymnt_plan ORDER BY tot DESC")
df_train_aux$pymnt_plan[df_train_aux$pymnt_plan == 'y'] = 1
df_train_aux$pymnt_plan[df_train_aux$pymnt_plan != 1] = 0
#Drop of features with many NA values
df_train_aux$addr_state <- NULL
df_train_aux$mths_since_last_delinq <- NULL
df_train_aux$mths_since_last_record <- NULL
## 7th step
#ZIP code treatment (get 3 digits and ignore the rest)
#df_train_aux$zip_code_3digit <- df_train_aux$zip_code
df_train_aux$zip_code_3digit <- sapply(df_train_aux$zip_code, function(x) as.numeric(substr(x, start = 0, stop = 3)))
sqldf("SELECT zip_code_3digit, count(zip_code_3digit) as tot FROM df_train_aux GROUP BY zip_code_3digit ORDER BY tot DESC")
df_train_aux$zip_code <- NULL
## 7th step
df_train_aux$collection_recovery_fee <- NULL   #drop of columns that are redundant with other consecutive fields
df_train_aux$last_pymnt_amnt <- NULL
df_train_aux$total_pymnt <- NULL
df_train_aux$out_prncp <- NULL
df_train_aux$mths_since_recent_bc_dlq <- NULL
#factor encoding --> last_pymnt_d
df_train_aux$fact_last_pymnt_d <- as.numeric(factor(df_train_aux$last_pymnt_d))
df_train_aux$last_pymnt_d <- NULL
df_train_aux$collections_12_mths_ex_med <- NULL
df_train_aux$mths_since_last_major_derog <-NULL
df_train_aux$annual_inc_joint <- NULL
df_train_aux$dti_joint <- NULL
df_train_aux$verification_status_joint <- NULL
df_train_aux$acc_now_delinq <- NULL
## 8th step
df_train_aux$chargeoff_within_12_mths <- NULL
df_train_aux$delinq_amnt <- NULL
df_train_aux$num_accts_ever_120_pd <- NULL
df_train_aux$num_bc_tl <- NULL
df_train_aux$num_rev_tl_bal_gt_0 <- NULL
df_train_aux$num_tl_120dpd_2m <- NULL
df_train_aux$num_tl_30dpd <- NULL
df_train_aux$il_util <- NULL
## 9th step
#conversion of Term and Grade to numerical from category
df_train_aux$fact_term <- as.numeric(factor(df_train_aux$term))
df_train_aux$term <- NULL
df_train_aux$fact_grade <- as.numeric(factor(df_train_aux$grade))
df_train_aux$grade <- NULL
#convert title into numerical
sqldf("SELECT title, count(title) as tot FROM df_train_aux GROUP BY title HAVING tot > 15 ORDER BY tot DESC")
df_train_aux$fact_title <- as.numeric(factor(df_train_aux$title))
df_train_aux$title <- NULL
#Convert this field into binary
sqldf("SELECT verification_status, count(verification_status) as tot FROM df_train_aux GROUP BY verification_status HAVING tot > 15 ORDER BY tot DESC")
df_train_aux$verification_status[df_train_aux$verification_status == "Source Verified" | 
                                 df_train_aux$verification_status == "Verified"] = 1
df_train_aux$verification_status[df_train_aux$verification_status != 1] = 0
#we leave purpose with no one-hot transformation, so with initial_list_status
#For date conversion, I'll get the year. Aux function to handle conversion errors
convert_date_ecl  <- function(str){
  date <- substr(str, start = 5, stop = 8)
  if (is.numeric(as.numeric(date))) {
    numb <- as.numeric(date)
  } else {
    numb <- 1900
  }
  return (numb)
}
#sapply to convert the date field into numeric year
df_train_aux$earliest_cr_line_year <- sapply(df_train_aux$earliest_cr_line, convert_date_ecl)
df_train_aux$earliest_cr_line_year[is.na(df_train_aux$earliest_cr_line_year)] = 1900   #to avoid NA (default value)
df_train_aux$earliest_cr_line <- NULL
#To pass the field to "> 2004" (1) or "< 2004" (0)
df_train_aux$earliest_cr_line_year[df_train_aux$earliest_cr_line_year >= 2004] = 1   #to convert to binary field
df_train_aux$earliest_cr_line_year[df_train_aux$earliest_cr_line_year != 1] = 0
df_train_aux$recoveries <- NULL
df_train_aux$last_credit_pull_d <- NULL
# 10th
# Final step is check target variable. No transformation is done if no NA are found
df_train_aux$target <- as.numeric(factor(df_train_aux$l_state))
df_train_aux$l_state <- NULL
sqldf("SELECT target, count(target) as tot FROM df_train_aux GROUP BY target HAVING tot > 15 ORDER BY tot DESC")   #ok, 2 values
colnames(df_train_aux)

##Basic treatment of NA; set to 0. Could also be set to mean or median if wanted
df_train_aux$annual_inc[is.na(df_train_aux$annual_inc)] = 0
df_train_aux$dti[is.na(df_train_aux$dti)] = 0
df_train_aux$open_acc[is.na(df_train_aux$open_acc)] = 0
df_train_aux$revol_bal[is.na(df_train_aux$revol_bal)] = 0
df_train_aux$revol_util[is.na(df_train_aux$revol_util)] = 0
df_train_aux$out_prncp_inv[is.na(df_train_aux$out_prncp_inv)] = 0
df_train_aux$tot_coll_amt[is.na(df_train_aux$tot_coll_amt)] = 0
df_train_aux$tot_cur_bal[is.na(df_train_aux$tot_cur_bal)] = 0
df_train_aux$all_util[is.na(df_train_aux$all_util)] = 0
df_train_aux$avg_cur_bal[is.na(df_train_aux$avg_cur_bal)] = 0
df_train_aux$bc_open_to_buy[is.na(df_train_aux$bc_open_to_buy)] = 0
df_train_aux$bc_util[is.na(df_train_aux$bc_util)] = 0
df_train_aux$mo_sin_old_rev_tl_op[is.na(df_train_aux$mo_sin_old_rev_tl_op)] = 0
df_train_aux$percent_bc_gt_75[is.na(df_train_aux$percent_bc_gt_75)] = 0
df_train_aux$pub_rec_bankruptcies[is.na(df_train_aux$pub_rec_bankruptcies)] = 0
df_train_aux$num_actv_bc_tl[is.na(df_train_aux$num_actv_bc_tl)] = 0
df_train_aux$num_actv_rev_tl[is.na(df_train_aux$num_actv_rev_tl)] = 0
df_train_aux$zip_code_3digit[is.na(df_train_aux$zip_code_3digit)] = 0
df_train_aux$inq_last_6mths[is.na(df_train_aux$inq_last_6mths)] = 0
df_train_aux$fact_purpose[is.na(df_train_aux$fact_purpose)] = 0
df_train_aux$mo_sin_old_il_acct[is.na(df_train_aux$mo_sin_old_il_acct)] = 0
df_train_aux$pub_rec[is.na(df_train_aux$pub_rec)] = 0
df_train_aux$mths_since_rcnt_il[is.na(df_train_aux$mths_since_rcnt_il)] = 0
df_train_aux$total_bal_il[is.na(df_train_aux$total_bal_il)] = 0
df_train_aux$open_rv_12m[is.na(df_train_aux$open_rv_12m)] = 0
df_train_aux$fact_application_type[is.na(df_train_aux$fact_application_type)] = 0
df_train_aux$open_rv_24m[is.na(df_train_aux$open_rv_24m)] = 0
###### No further NA values are detected

##Two columns in df have to be converted to numeric
df_train_aux$verification_status <- sapply(df_train_aux$verification_status, as.numeric)
df_train_aux$pymnt_plan <- sapply(df_train_aux$pymnt_plan, as.numeric)

#df_train_aux <- apply(df_train_aux, 2, function(x){replace(x, is.na(x), 0)})
aux <- df_train_aux[!complete.cases(df_train_aux), ]

######################################################
#EDA
######################################################
#Correlations and scatter plots
corr_risk <- cor(df_train_aux[, 2:30])
corrplot(corr_risk, method = "square")
#In this kind of graphic, we can see that there are features that are, somehow, a transformation from others.
#All those in intense "blue" or "red" outside diagonal (self) show strong correlation. Scatters can also show this:
pairs(df_train_aux[1:1000,1:10])    #graphic is computationally expensive, I give it just to show (corrplot is enough)

#We now show an example of how these features should be chequed within a further analysis, to check distribution,
#likelyhood of having strong outliers, etc. I'll show it for 16 variables only (histograms and boxplot)
ggplot(data = melt(df_train_aux[,2:17]), mapping = aes(x = value)) + geom_histogram(bins = 20) +
  facet_wrap(~variable, scales = 'free_x')
ggplot(data = melt(df_train_aux[,18:33]), mapping = aes(x = value)) + geom_histogram(bins = 20) +
  facet_wrap(~variable, scales = 'free_x')


#####################################################
## Generalization (function) of feature engineering
## Pass to functionn of all the steps done
#####################################################
process_data_risk_model_smava <- function(df_in) {
  #1st step: we drop a number of columns that are useless due to emptyness. We create df_in 
  df_aux <- df_in[, -c(2,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135)]
  #2nd step: row with invalid data removal. Through filtering NA data in the column "loan_amnt"
  df_aux <- subset(df_aux, !is.na(df_aux$loan_amnt))
  #3rd step: additional features have to be deleted (NA, high frequency of erroneous data, sparse / useless features, redundant) 
  # and check on other fields to see their different (unique) values
  fields_to_process <- c(15,16,17,18,19,20,24,34,35,26,44,45,46,50,51,84,85,86,87)
  #Unique values checks
  #We must transform these three fields in order to use them
  ## --> purpose
  df_aux$purpose[df_aux$purpose != "debt_consolidation" & df_aux$purpose != "credit_card" & 
                         df_aux$purpose != "home_improvement"] = "other"
  df_aux$fact_purpose <- as.numeric(factor(df_aux$purpose))   #convert into numeric (one per category)
  ## --> initial_list_status
  df_aux$initial_list_status[df_aux$initial_list_status != "w" & df_aux$initial_list_status != "f"] = "other"
  df_aux$fact_initial_list_status <- as.numeric(factor(df_aux$initial_list_status))
  ## --> application_type
  df_aux$application_type[df_aux$application_type != "INDIVIDUAL" & df_aux$application_type != "JOINT"] = "OTHER"
  df_aux$fact_application_type <- as.numeric(factor(df_aux$application_type))
  #Columns drop
  fields_to_process <- c(17,18,20,24,44,46,50,84,85,86,87)
  df_aux <- df_aux[,-fields_to_process]
  df_aux$purpose <- NULL
  df_aux$initial_list_status <- NULL
  df_aux$application_type <- NULL
  ## 4th step
  df_aux$funded_amnt <- NULL
  df_aux$funded_amnt_inv <- NULL
  df_aux$sub_grade <- NULL
  ## 5th step
  df_aux$emp_title <- NULL    #4 features to be dropped (very disperse categories)
  df_aux$emp_length <- NULL
  df_aux$home_ownership <- NULL
  df_aux$issue_d <- NULL
  ## 6th step
  #We are going to transform and aggregate Payment plan field
  df_aux$pymnt_plan[df_aux$pymnt_plan == 'y'] = 1
  df_aux$pymnt_plan[df_aux$pymnt_plan != 1] = 0
  #Drop of features with many NA values
  df_aux$addr_state <- NULL
  df_aux$mths_since_last_delinq <- NULL
  df_aux$mths_since_last_record <- NULL
  ## 7th step
  #ZIP code treatment (get 3 digits and ignore the rest)
  #df_aux$zip_code_3digit <- df_aux$zip_code
  df_aux$zip_code_3digit <- sapply(df_aux$zip_code, function(x) as.numeric(substr(x, start = 0, stop = 3)))
  df_aux$zip_code <- NULL
  ## 7th step
  df_aux$collection_recovery_fee <- NULL   #drop of columns that are redundant with other consecutive fields
  df_aux$last_pymnt_amnt <- NULL
  df_aux$total_pymnt <- NULL
  df_aux$out_prncp <- NULL
  df_aux$mths_since_recent_bc_dlq <- NULL
  #factor encoding --> last_pymnt_d
  #df_aux$last_pymnt_d
  #df_aux$fact_last_pymnt_d <- as.numeric(factor(df_aux$last_pymnt_d))
  df_aux$last_pymnt_d <- NULL
  df_aux$collections_12_mths_ex_med <- NULL
  df_aux$mths_since_last_major_derog <-NULL
  df_aux$annual_inc_joint <- NULL
  df_aux$dti_joint <- NULL
  df_aux$verification_status_joint <- NULL
  df_aux$acc_now_delinq <- NULL
  ## 8th step
  df_aux$chargeoff_within_12_mths <- NULL
  df_aux$delinq_amnt <- NULL
  df_aux$num_accts_ever_120_pd <- NULL
  df_aux$num_bc_tl <- NULL
  df_aux$num_rev_tl_bal_gt_0 <- NULL
  df_aux$num_tl_120dpd_2m <- NULL
  df_aux$num_tl_30dpd <- NULL
  df_aux$il_util <- NULL
  ## 9th step
  #conversion of Term and Grade to numerical from category
  df_aux$fact_term <- as.numeric(factor(df_aux$term))
  df_aux$term <- NULL
  df_aux$fact_grade <- as.numeric(factor(df_aux$grade))
  df_aux$grade <- NULL
  #convert title into numerical
  #df_aux$fact_title <- as.numeric(factor(df_aux$title))
  df_aux$title <- NULL
  #Convert this field into binary
  df_aux$verification_status[df_aux$verification_status == "Source Verified" | 
                                     df_aux$verification_status == "Verified"] = 1
  df_aux$verification_status[df_aux$verification_status != 1] = 0
  #we leave purpose with no one-hot transformation, so with initial_list_status
  #sapply to convert the date field into numeric year
  df_aux$earliest_cr_line_year <- sapply(df_aux$earliest_cr_line, convert_date_ecl)
  df_aux$earliest_cr_line_year[is.na(df_aux$earliest_cr_line_year)] = 1900   #to avoid NA (default value)
  df_aux$earliest_cr_line <- NULL
  #To pass the field to "> 2004" (1) or "< 2004" (0)
  df_aux$earliest_cr_line_year[df_aux$earliest_cr_line_year >= 2004] = 1   #to convert to binary field
  df_aux$earliest_cr_line_year[df_aux$earliest_cr_line_year != 1] = 0
  df_aux$recoveries <- NULL
  df_aux$last_credit_pull_d <- NULL
  # 10th
  # Final step is check target variable. No transformation is done if no NA are found
  df_aux$target <- as.numeric(factor(df_aux$l_state))
  df_aux$l_state <- NULL

  ##Basic treatment of NA; set to 0. Could also be set to mean or median if wanted
  df_aux$annual_inc[is.na(df_aux$annual_inc)] = 0
  df_aux$dti[is.na(df_aux$dti)] = 0
  df_aux$open_acc[is.na(df_aux$open_acc)] = 0
  df_aux$revol_bal[is.na(df_aux$revol_bal)] = 0
  df_aux$revol_util[is.na(df_aux$revol_util)] = 0
  df_aux$out_prncp_inv[is.na(df_aux$out_prncp_inv)] = 0
  df_aux$tot_coll_amt[is.na(df_aux$tot_coll_amt)] = 0
  df_aux$tot_cur_bal[is.na(df_aux$tot_cur_bal)] = 0
  df_aux$all_util[is.na(df_aux$all_util)] = 0
  df_aux$avg_cur_bal[is.na(df_aux$avg_cur_bal)] = 0
  df_aux$bc_open_to_buy[is.na(df_aux$bc_open_to_buy)] = 0
  df_aux$bc_util[is.na(df_aux$bc_util)] = 0
  df_aux$mo_sin_old_rev_tl_op[is.na(df_aux$mo_sin_old_rev_tl_op)] = 0
  df_aux$percent_bc_gt_75[is.na(df_aux$percent_bc_gt_75)] = 0
  df_aux$pub_rec_bankruptcies[is.na(df_aux$pub_rec_bankruptcies)] = 0
  df_aux$num_actv_bc_tl[is.na(df_aux$num_actv_bc_tl)] = 0
  df_aux$num_actv_rev_tl[is.na(df_aux$num_actv_rev_tl)] = 0
  df_aux$zip_code_3digit[is.na(df_aux$zip_code_3digit)] = 0
  df_aux$inq_last_6mths[is.na(df_aux$inq_last_6mths)] = 0
  df_aux$fact_purpose[is.na(df_aux$fact_purpose)] = 0
  df_aux$mo_sin_old_il_acct[is.na(df_aux$mo_sin_old_il_acct)] = 0
  df_aux$pub_rec[is.na(df_aux$pub_rec)] = 0
  df_aux$mths_since_rcnt_il[is.na(df_aux$mths_since_rcnt_il)] = 0
  df_aux$total_bal_il[is.na(df_aux$total_bal_il)] = 0
  df_aux$open_rv_12m[is.na(df_aux$open_rv_12m)] = 0
  df_aux$fact_application_type[is.na(df_aux$fact_application_type)] = 0
  df_aux$open_rv_24m[is.na(df_aux$open_rv_24m)] = 0
  df_aux$loan_amnt[is.na(df_aux$loan_amnt)] = 0
  ###### No further NA values are detected
  
  ##Two columns in df have to be converted to numeric
  df_aux$verification_status <- sapply(df_aux$verification_status, as.numeric)
  df_aux$pymnt_plan <- sapply(df_aux$pymnt_plan, as.numeric)
  
  return (df_aux)
}

#Load of train / test Dataset, and for target dataset to use the model with
df_train <- read.csv(file = 'RAcredit_train.csv', header = TRUE, sep = ',', stringsAsFactors = FALSE)
df_target <- read.csv(file = 'RAcredit_test.csv', header = TRUE, sep = ',', stringsAsFactors = FALSE)
## 75% of the sample size
smp_size <- floor(0.75 * nrow(df_train))

## set the seed to make your partition reproductible
#set.seed(123)
train_ind <- sample(seq_len(nrow(df_train)), size = smp_size)

df_train_split <- df_train[train_ind, ]
df_test_split <- df_train[-train_ind, ]

#transformation
df_train_aux <- process_data_risk_model_smava(df_train_split)
df_test_aux <- process_data_risk_model_smava(df_test_split)
df_train_split <- NULL
df_test_split <- NULL



#######################################################################################################
## Machine Learning Model (Logistic Regression)
#######################################################################################################
# use glm to train the model on the training dataset. make sure to set family to "binomial"

df_train_aux$target = df_train_aux$target - 1   #to convert to full binary (0-1, instead of 1-2)
df_test_aux$target = df_test_aux$target - 1
#'pymnt_plan', 'revol_bal', 'total_rec_prncp', 'tot_cur_bal'
mod_fit_one <- glm(target ~ pymnt_plan + revol_bal + total_rec_prncp + tot_cur_bal,
                   data=df_train_aux, family="binomial")
summary(mod_fit_one)
anova(mod_fit_one, test = "Chisq")

mod_fit_comp <- glm(target ~ loan_amnt+int_rate+installment+inq_last_6mths+verification_status+pymnt_plan+dti+inq_last_6mths+
                   open_acc+pub_rec+revol_bal+revol_util+total_acc+out_prncp_inv+total_pymnt_inv+total_rec_prncp+total_rec_int+
                   total_rec_late_fee+tot_coll_amt+total_rec_prncp+total_rec_late_fee+tot_coll_amt+tot_cur_bal+open_acc_6m+
                   open_il_6m+open_il_12m+open_il_24m+mths_since_rcnt_il+total_bal_il+open_rv_12m+open_rv_24m+max_bal_bc+
                   all_util+total_rev_hi_lim+inq_fi+total_cu_tl+inq_last_12m+acc_open_past_24mths+avg_cur_bal+bc_open_to_buy+
                   bc_util+mo_sin_old_il_acct+mo_sin_old_rev_tl_op+mo_sin_rcnt_rev_tl_op+mo_sin_rcnt_tl+mort_acc+num_actv_bc_tl+
                   num_actv_rev_tl+num_bc_sats+num_tl_90g_dpd_24m+num_tl_op_past_12m+pct_tl_nvr_dlq+percent_bc_gt_75+
                   pub_rec_bankruptcies+tax_liens+tot_hi_cred_lim+total_bal_ex_mort+total_bc_limit+total_il_high_credit_limit+
                   fact_purpose+fact_initial_list_status+fact_application_type+zip_code_3digit+fact_term+fact_grade+
                   earliest_cr_line_year, 
                   data=df_train_aux, family="binomial")

summary(mod_fit_comp) # estimates. Al p-values are under 0.05 (conf. level 95%), including Intercept (constant)
anova(mod_fit_comp, test = "Chisq")
#comparison of the two models
lrtest(mod_fit_one, mod_fit_comp)

#exp(coef(mod_fit_comp)) # odds ratios
df_test_aux$predict <- predict(mod_fit_one, newdata=df_test_aux, type="response") # predicted probabilities
df_test_aux$predict <- sapply(y_pred, function(x) as.numeric(round(x, digits = 0)))
summary(df_test_aux$predict)
df_test_aux$dif_target = abs(df_test_aux$target - df_test_aux$predict)

#We calculate the percentage or success / fail on each catagory: 
# Fully Paid ok % and ko %
# Default ok % and ko %
tot_fp_ok <- length(df_test_aux$target[df_test_aux$target == 1 & df_test_aux$predict == 1])
tot_fp_ko <- length(df_test_aux$target[df_test_aux$target == 1 & df_test_aux$predict == 0])
tot_df_ok <- length(df_test_aux$target[df_test_aux$target == 0 & df_test_aux$predict == 0])
tot_df_ko <- length(df_test_aux$target[df_test_aux$target == 0 & df_test_aux$predict == 1])
print("% Ok for Fully Paid credits: ")
print(tot_fp_ok * 100 / (tot_fp_ok+tot_fp_ko))
print("% Ok for Default credits: ")
print(tot_df_ok * 100 / (tot_df_ok+tot_df_ko))

########## Second model (all variables)
df_test_aux$predict <- predict(mod_fit_comp, newdata=df_test_aux, type="response") # predicted probabilities
df_test_aux$predict <- sapply(y_pred, function(x) as.numeric(round(x, digits = 0)))
summary(df_test_aux$predict)
df_test_aux$dif_target = abs(df_test_aux$target - df_test_aux$predict)

#We calculate the percentage or success / fail on each catagory: 
# Fully Paid ok % and ko %
# Default ok % and ko %
tot_fp_ok <- length(df_test_aux$target[df_test_aux$target == 1 & df_test_aux$predict == 1])
tot_fp_ko <- length(df_test_aux$target[df_test_aux$target == 1 & df_test_aux$predict == 0])
tot_df_ok <- length(df_test_aux$target[df_test_aux$target == 0 & df_test_aux$predict == 0])
tot_df_ko <- length(df_test_aux$target[df_test_aux$target == 0 & df_test_aux$predict == 1])
print("% Ok for Fully Paid credits: ")
print(tot_fp_ok * 100 / (tot_fp_ok+tot_fp_ko))
print("% Ok for Default credits: ")
print(tot_df_ok * 100 / (tot_df_ok+tot_df_ko))


#########################################################################################
## Target (test) dataset model run
#########################################################################################
# We apply the model (the first one, as there is no significant differencts between them) to the target dataset
df_target_aux <- process_data_risk_model_smava(df_target)
df_target_aux$target <- predict(mod_fit_one, newdata=df_target_aux, type="response") # predicted probabilities
print("Total elements on target dataset:")
print(length(df_target_aux$target))
print("Total predictions (Fully Paid):")
print(length(df_target_aux$target[df_target_aux$target > 0.5]))
print("Total predictions (Default):")
print(length(df_target_aux$target[df_target_aux$target <= 0.5]))

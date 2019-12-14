# Preparation: importing packages and data
pacman::p_load(tidyverse, data.table, Metrics, lightgbm)
sample_submission = fread("Data/SampleSubmission.csv")
train = fread("Data/Train.csv")
test = fread("Data/Test.csv")

test$target = NA # add target column to test data before merging data
merged_dat = as.data.frame(rbind(train, test)) 

# Label Encoding
char_vars = names(select_if(merged_dat, is.character))
merged_dat[, char_vars] = apply(merged_dat[, char_vars], 2, function(x) as.numeric(as.factor(x))-1 )


# Data Split
train_dat = merged_dat[!is.na(merged_dat$target), ]
test_dat =  merged_dat[is.na(merged_dat$target), ]

# LightGBM Modeling
lgbm_train = lgb.Dataset(data = as.matrix(select(train_dat, -target)), label = train_dat$target)
lbgm_test = data.matrix(select(test_dat, -target))

params <- list(boosting_type = 'gbdt', 
               objective = "regression" , 
               metric = "rmse",
               boost_from_average = "true", 
               learning_rate = 0.008, 
               num_leaves = 400, 
               min_gain_to_split = 0,
               feature_fraction = 0.7, 
               bagging_freq = 1,
               bagging_fraction = 0.7,
               min_data_in_leaf = 200,
               lambda_l1 = 0,
               lambda_l2 = 0)
lgb.model <- lgb.train(params = params,
                       data = lgbm_train,
                       nrounds=5000, 
                       verbose=1,
                       eval_freq=100)
lgbm_pred = predict(lgb.model, as.matrix(lbgm_test)) 


# Make & export a submission file
sample_submission$target = lgbm_pred
write.csv(sample_submission, "submission.csv", row.names = F)




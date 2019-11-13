#RF Model-----------------------------------------------------------------------------------------

library(dplyr)
final= read.csv("/Users/devarshipancholi/Desktop/df3_1.csv")

#str(final)

dataset <- select(final,
                  code_module,code_presentation,gender,
                  region,highest_education,imd_band,
                  studied_credits,num_of_prev_attempts,
                  age_band,disability,sum_click,assessment_type,score,weight,final_result)

target <- ("final_result")
dependent <- (names(dataset)[names(dataset) != target])

library(ROSE)

set.seed(1234)
split <- (.70)
library (caret)
library(kernlab)
library(xgboost)
index <- createDataPartition(dataset$final_result, p=split, list=FALSE)

train.df <- dataset[ index,]
library(DMwR)
train_smote <- SMOTE( final_result ~ ., train.df, perc.over = 100, perc.under = 200)
prop.table(table(train_smote$final_result))
train.df = train_smote
test.df <- dataset[ -index,]

#rf <- train(train.df[,dependent],train.df[,target], method='rf')
#summary(rf)

library(randomForest)
rf <- randomForest(final_result~., data = train.df,ntree = 5000,na.action=na.omit)
summary(rf)
rf.predict <- predict(rf,test.df[,dependent])
r <- data.frame(Actual = test.df$final_result , Prediction = rf.predict)
r <- table(r)
r

accuracy <- (r[1,1] + r[2,2])/sum(r)
accuracy

precision <- (r[2,2]/(r[2,2] + r[1,2]))
precision

recall <- (r[2,2]/(r[2,2] + r[2,1]))
recall

f_score <- 2*((precision*recall)/(precision+recall))
f_score

g_score <- sqrt(precision*recall)
g_score

library(pROC)
library(caret)

#rf.probs <- predict(rf,test.df[,dependent],type="prob") 
#rf.plot<-plot(roc(test.df$final_result,rf.probs[,2]))
#rf.plot<-lines(roc(test.df$final_result,rf.probs[,2]), col="blue")

confusionMatrix(rf.predict,test.df[,target], positive = "Pass")
multiclass.roc(test.df$final_result, predict(rf, test.df[,dependent], type= "prob", percent=FALSE))

--------------------------------------------------------------------------------------------------

#GBM Model
  
fitControl <- trainControl(method = "cv", number = 20, sampling = "up", classProbs = TRUE)

#lm <- (train(train.df[,dependent],train.df[,target], method='glm'))
gbm <- train(train.df[,dependent],train.df[,target], method='gbm', trControl = fitControl)
summary(gbm)
  
gbm.predict <- predict(gbm,test.df[,dependent],type="raw")
#lm.predict <- predict(lm,test.df[,dependent],type="raw")
#summary(gbm.predict)

r <- data.frame(Actual = test.df$final_result , Prediction = gbm.predict)
r <- table(r)
r

accuracy <- (r[1,1] + r[2,2])/sum(r)
accuracy

precision <- (r[2,2]/(r[2,2] + r[1,2]))
precision

recall <- (r[2,2]/(r[2,2] + r[2,1]))
recall

f_score <- 2*((precision*recall)/(precision+recall))
f_score

g_score <- sqrt(precision*recall)
g_score

library(pROC)

gbm.probs <- predict(gbm,test.df[,dependent],type="prob")    
#rf.probs <- predict(rf,test.df[,dependent],type="prob") 

gbm.plot<-plot(roc(test.df$final_result,gbm.probs[,2]))
#rf.plot<-lines(roc(test.df$final_result,rf.probs[,2]), col="blue")

confusionMatrix(gbm.predict,test.df[,target], positive = "Pass")
multiclass.roc(test.df$final_result, predict(gbm, test.df[,dependent], type= "prob", percent=FALSE))

---------------------------------------------------------------------------------------------------

#SVM Model

library(e1071)
library(pROC)
svm = svm(formula = final_result ~ .,
           data = train.df,
           type = 'C-classification',
           kernel = 'radial')

svm.predict <- predict(svm,test.df[,dependent], probability = TRUE)
confusionMatrix(svm.predict,test.df[,target])
r <- data.frame(Actual = test.df$final_result , Prediction = svm.predict)
r <- table(r)
r

accuracy <- (r[1,1] + r[2,2])/sum(r)
accuracy

precision <- (r[2,2]/(r[2,2] + r[1,2]))
precision

recall <- (r[2,2]/(r[2,2] + r[2,1]))
recall

f_score <- 2*((precision*recall)/(precision+recall))
f_score

g_score <- sqrt(precision*recall)
g_score

multiclass.roc(test.df$final_result, predict(svm, test.df[,dependent], type= "prob", percent=FALSE))

roc_obj <- roc(test.df$final_result, predict(dt, test.df[,dependent], type= "vector", percent=FALSE))
auc(roc_obj)

svm.probs <- predict(svm,test.df[,dependent],type="prob")    
svm.probs 

------------------------------------------------------------------------------------------------------------------
  
#DT MODEL

library(rpart) 
library(rpart.plot) 

dt <-rpart(final_result ~.,data = train.df, method = "class")
dt
#printcp(studentTree)
#rpart.plot(studentTree)

dt.predict <-predict(dt,test.df,type="class")
test.df$final_result<-as.factor(test.df1$final_result)
require(caret)
confusionMatrix(dt.predict,test.df$final_result)
r <- data.frame(Actual = test.df$final_result , Prediction = dt.predict)
r <- table(r)
r
precision <- (r[2,2]/(r[2,2] + r[1,2]))
precision

multiclass.roc(test.df$final_result, predict(dt, test.df[,dependent], type= "prob", percent=FALSE))
               
----------------------------------------------------------------------------------------------------

#Bayes Model
  
library(e1071)
  
bayes <-naiveBayes(final_result ~.,data = train.df, control=rpart.control(minsplit=2, minbucket=1, cp=0.001))
bayes

bayes.predict <- predict(bayes,test.df)
test.df$final_result <- as.factor(test.df$final_result)
bayes.predict <-as.factor(bayes.predict)

require(caret)
confusionMatrix(bayes.predict,test.df$final_result)
r <- data.frame(Actual = test.df$final_result , Prediction = bayes.predict)
r <- table(r)
r
precision <- (r[2,2]/(r[2,2] + r[1,2]))
precision           
               
multiclass.roc(test.df$final_result, predict(bayes, test.df[,dependent], type= "raw", percent=FALSE))
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

rf <- train(train.df[,dependent],train.df[,target], method='rf')
summary(rf)

library(randomForest)
rf <- randomForest(final_result~., data = train.df,ntree = 5000,na.action=na.omit)
summary(rf)
rf.predict <- predict(rf,test.df[,dependent],type="response")
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


#Intsalling necessary libraries and loading dataset
library(ranger)
library(caret)
library(data.table)
creditcard_data <- read.csv("creditcard.csv")

#Basic viewing of structure and data contents
dim(creditcard_data)
head(creditcard_data,6)

table(creditcard_data$Class)
summary(creditcard_data$Amount)
names(creditcard_data)
var(creditcard_data$Amount)

sd(creditcard_data$Amount)

#Scaling data
creditcard_data$Amount=scale(creditcard_data$Amount)
NewData=creditcard_data[,-c(1)]
head(NewData)

#Splitting data into train and test data 
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)

#Building a decision tree on the training data and predicting the output of test data
#using the same.
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , train_data, method = 'class')
predicted_val <- predict(decisionTree_model, test_data, type = 'class')
probability <- predict(decisionTree_model, test_data, type = 'prob')
rpart.plot(decisionTree_model)

predict_unseen <-predict(decisionTree_model, test_data, type = 'class')

table_mat <- table(test_data$Class, predicted_val)
table_mat

#Testing for accuracy
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)

print(paste('Accuracy for test', accuracy_Test))
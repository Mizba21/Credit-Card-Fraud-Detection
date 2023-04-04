# Importing data set and viewing it
df=read.csv("C:/Users/Mizba/Downloads/creditcard.csv",stringsAsFactors=T)
df=na.omit(df)
table(head(df))
dim(df)
names(df)

# Installation of required packages
install.packages("e1071")
install.packages("caTools")
install.packages("caret")

library(e1071)
library(caTools)
library(caret)


# Splitting data into training and testing data
split <- sample.split(df, SplitRatio = 0.7)
train_cl <- subset(df, split == "TRUE")
test_cl <- subset(df, split == "FALSE")

# Fitting the training data to a Naive Bayes Model
set.seed(120)  # Setting Seed
classifier_cl <- naiveBayes(Class ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27+V28, data = train_cl)
classifier_cl


# Predicting on test data'
y_pred <- predict(classifier_cl, newdata = test_cl)
y_pred

#Accuracy using F-measure (best for unbalanced classification)
install.packages("MLmetrics")
library("MLmetrics")
F1_Score(test_cl$Class, y_pred)*100

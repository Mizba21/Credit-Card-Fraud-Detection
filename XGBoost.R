library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(rpart)# for decision tree model
library(xgboost) # for xgboost model


df = read.csv('creditcard.csv')
head(df)
summary(df)
# checking class imbalance
table(df$Class)
# class imbalance in percentage
prop.table(table(df$Class))



# function to set plot height and width
fig <- function(width, heigth){
  options(repr.plot.width = width, repr.plot.height = heigth)
}

#We observe that most of the data features are not correlated. This is because before publishing, 
#most of the features were presented to a Principal Component Analysis (PCA) algorithm. 
#The features V1 to V28 are most probably the Principal Components resulted after propagating the real features 
#through PCA. We do not know if the numbering of the features reflects the 
#importance of the Principal Components.
fig(14, 8)
correlations <- cor(df[,-1],method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")




#DATA PREPARATION


#‘Time’ feature does not indicate the actual time of the transaction and 
#is more of listing the data in chronological order.we assume that ‘Time’ feature has little or no
#significance in correctly classifying a fraud transaction and hence eliminate this 
#column from further analysis.

#Remove 'Time' variable
df <- df[,-1]


#Change 'Class' variable to factor
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")

#Scale numeric variables

df[,-30] <- scale(df[,-30])

head(df)

#Split the data into train and test data sets
set.seed(123)
split <- sample.split(df$Class, SplitRatio = 0.7)
train <-  subset(df, split == TRUE)
test <- subset(df, split == FALSE)


# class ratio initially
table(train$Class)


# downsampling
set.seed(9560)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
table(down_train$Class)


# upsampling
set.seed(9560)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
table(up_train$Class)

#XGBOOST ALGORITHM

# Convert class labels from factor to numeric

labels <- up_train$Class

y <- recode(labels, 'Not_Fraud' = 0, "Fraud" = 1)

set.seed(42)
xgb <- xgboost(data = data.matrix(up_train[,-30]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)



xgb_pred <- predict(xgb, data.matrix(test[,-30]))

roc.curve(test$Class, xgb_pred, plotit = TRUE)



#Take a look at the important features here
names <- dimnames(data.matrix(up_train[,-30]))[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])


#As the dataset is an imbalanced one ,the accuracy won't make more sense ,hence we have to calculate the
# AUCC SCORE
#With an auc score of 0.977 the XGBOOST model has performed the best






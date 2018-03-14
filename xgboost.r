#Library
# install.packages("xgboost")
# install.packages("Hmisc")
# library(xgboost)
require(xgboost)
library(Matrix)
library(Hmisc)

# 1.Data Exploration& Data Cleaning
#read the data
train<- read.csv("D:/RWorkSpaces/Data/train_plia.csv")
dim(train)
str(train)

#creater feature vector
categoricalFeatures<- c("Product_Info_1", "Product_Info_2", "Product_Info_3",
                         "Product_Info_5", "Product_Info_6", "Product_Info_7",
                         "Employment_Info_2", "Employment_Info_3", "Employment_Info_5",
                         "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3",
                         "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
                         "Insurance_History_1", "Insurance_History_2", "Insurance_History_3",
                         "Insurance_History_4", "Insurance_History_7", "Insurance_History_8",
                         "Insurance_History_9", "Family_Hist_1", "Medical_History_2",
                         "Medical_History_3", "Medical_History_4", "Medical_History_5",
                         "Medical_History_6", "Medical_History_7", "Medical_History_8",
                         "Medical_History_9", "Medical_History_11", "Medical_History_12",
                         "Medical_History_13", "Medical_History_14", "Medical_History_16",
                         "Medical_History_17", "Medical_History_18", "Medical_History_19",
                         "Medical_History_20", "Medical_History_21", "Medical_History_22",
                         "Medical_History_23", "Medical_History_25", "Medical_History_26",
                         "Medical_History_27", "Medical_History_28", "Medical_History_29",
                         "Medical_History_30", "Medical_History_31", "Medical_History_33",
                         "Medical_History_34", "Medical_History_35", "Medical_History_36",
                         "Medical_History_37", "Medical_History_38", "Medical_History_39",
                         "Medical_History_40", "Medical_History_41")
continuousFeatures <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1",
                         "Employment_Info_4", "Employment_Info_6", "Insurance_History_5",
                         "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")
discreteFeatures <- c("Medical_History_1", "Medical_History_10", "Medical_History_15",
                       "Medical_History_24", "Medical_History_32")
# type conversion
for (feature in categoricalFeatures) {
  train[,feature]<- as.factor(train[,feature])
}

for (feature in continuousFeatures) {
  #replace missing values with the median of this feature
  train[is.na(train[,feature]),feature]<- median(train[, feature], na.rm = TRUE)
  train[,feature]<- as.numeric(train[,feature])
}

modeFuction<- function(x){
  x<- x[!is.na(x)]
  ux<- unique(x)
  ux[which.max(tabulate(match(x,ux)))]
}

for (feature in discreteFeatures) {
  #replace missing values with the mode of this feature
  train[is.na(train[,feature]),feature]<- modeFuction(train[,feature])
  train[,feature]<- as.numeric(train[,feature])
}

table(train$Response)
hist(train$Response)

#divide the data 
set.seed(666666666)
sample_size<- round(nrow(train)*0.3)
testIndex<- sample(1:nrow(train),sample_size)
# trainIndex<- setdiff(1:nrow(train),testIndex)
#change the data to the matirx
matrixData<- sparse.model.matrix(Response ~. -1 - Id,train)


test4xgb<- matrixData[testIndex,]
train4xgb<- matrixData[-testIndex,]

trainDM<- xgb.DMatrix(data=train4xgb, label=train[-testIndex,"Response"])


# Feature enginnering: choose the most important features




#Build the model
#setting the model params
xgbParams<- list(
  # booster which booster to use, can be gbtree or gblinear
  booster="gbtree",
  #eta control the learning rate: scale the contribution of each tree by a factor of 0 < eta < 1
  eta=0.01,
  verbose=1,
  # minimum loss reduction required to make a further partition on a leaf node of the tree
  gamma=0.01,
  # maximum depth of a tree
  max_depth=10,
  # Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees and this will prevent overfitting
  subsample=0.7,
  # subsample ratio of columns when constructing each tree
  colsample_bytree=0.5,
  # objective specify the learning task and the corresponding learning objective
  objective="reg:linear"
)




xgb_model<- xgb.train(params = xgbParams,data = trainDM,nrounds = 120)

#predict with the model
pred<- predict(xgb_model,test4xgb)
hist(pred)

cut_points <- seq(1.5, 7.5, by = 0.5)
train_pred_cut <- as.numeric(cut2(pred, c(-Inf, cut_points, Inf)))
hist(cut_points)

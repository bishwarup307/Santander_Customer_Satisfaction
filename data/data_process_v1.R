require(readr)
require(Hmisc)
require(dplyr)
require(caret)
require(Rtsne)

setwd("F:/Kaggle/Santander")
set.seed(1)

train <- read.csv("./RawData/train.csv")
test <- read.csv("./RawData/test.csv")

train$trainFlag <- 1
test$trainFlag <- 0
test$TARGET <- NA

alldata <- rbind(train, test)

const.cols <- names(which(sapply(alldata[, -c(1, 371, 372)], function(x) length(unique(x))) == 1))
alldata <- alldata[, !names(alldata) %in% const.cols] # remove constant cols

##### Removing identical features
features_pair <- combn(names(alldata), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(alldata[[f1]] == alldata[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

alldata <- alldata[, !names(alldata) %in% toRemove] # remove duplicate features

####### remove linear combos

#nzv <- nearZeroVar(alldata[, -c(1, 310, 311)], saveMetrics = TRUE)
#nearZeroVar.cols <- row.names(nzv[nzv$nzv == TRUE,])

lin.comb <- findLinearCombos(alldata[, -c(1, 310, 311)])
lin.comb.remove <- names(alldata[, -c(1, 310, 311)])[lin.comb$remove]
alldata <- alldata[, !names(alldata) %in% lin.comb.remove]

##### number of zeros in each row, 
alldata$zeroCount <- apply(alldata[, !names(alldata) %in% c("ID", "TARGET", "trainFlag")], 1, function(x) sum(x == 0))
alldata$rowSum <- apply(alldata[, !names(alldata) %in% c("ID", "TARGET", "trainFlag")], 1, function(x) sum(x, na.rm = TRUE))

###### binary cols
binCols <- names(which(sapply(alldata[, -c(1, 248:251)], function(x) length(unique(x)) == 2)))
binDF <- alldata[, binCols]


tsne <- Rtsne(as.matrix(binDF), dims = 2, perplexity = 30, check_duplicates = FALSE, pca = FALSE, theta = 0.5, verbose = TRUE)
tsneDF <- as.data.frame(tsne$Y)
names(tsneDF) <- c("tsne_f1", "tsne_f2")

alldata <- cbind(alldata, tsneDF)

# tsneDF <- alldata[, c("ID", "tsne_f1", "tsne_f2")]
# write_csv(tsneDF, "./RawData/tsne_features.csv")

###########################

save(alldata, file = "./data_V1/alldata_v1.RData")


######### get feature importance ########
ptr <- alldata[alldata$trainFlag == 1,]
pte <- alldata[alldata$trainFlag == 0,]

require(xgboost)

param <- list(objective = "binary:logistic",
              max_depth = 5,
              eta = 0.01,
              subsample = 0.8,
              colsample_bytree = 0.45,
              min_child_weight = 3,
              eval_metric = "auc")
feature.names <- names(ptr)[!names(ptr) %in% c("ID", "TARGET", "trainFlag")]
dtrain <- xgb.DMatrix(data = data.matrix(ptr[, feature.names]), label = ptr$TARGET)
watchlist <- list(train = dtrain)
clf <- xgb.train(params = param,
                 data = dtrain,
                 nround = 700,
                 print.every.n = 100,
                 watchlist = watchlist)

imp <- xgb.importance(feature.names, model = clf)
write_csv(imp, "./data_V1/feature_importance.csv")
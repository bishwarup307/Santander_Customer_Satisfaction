require(readr)
require(data.table)
require(bit64)
require(mRMRe)
require(caret)
require(xgboost)
require(Matrix)

dir <- 'F:/Kaggle/Santander/'
dataDir <- "./RawData/"
setwd(dir)
set.seed(10)

train <- as.data.frame(fread(paste0(dataDir, "train.csv")))
test <- as.data.frame(fread(paste0(dataDir, "test.csv")))
imp <- read.csv('./data_V1/feature_importance.csv')
test$TARGET <- NA
alldata <- rbind(train, test)

# remove constant columns
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
rm(train, test)
train <- alldata[which(!is.na(alldata$TARGET)),]
test <- alldata[which(is.na(alldata$TARGET)),]
feats <- setdiff(names(train), c("ID", "TARGET"))

# minimum redundancy maximum relevance feature selection 
trF <- train[, -1]
for(f in names(trF)){
  trF[[f]] <- as.numeric(trF[[f]])
}
X <- mRMR.data(data = trF)
fnm <- featureNames(X)
ss <- mRMR.classic(feature_count = 160,data = X, target_indices = c(309))
selectedFeats <- fnm[as.numeric(ss@filters$`309`)]

leftOuts <- setdiff(imp$Feature, selectedFeats)
# trF <- train[, c("TARGET", selectedFeats)]
# for(f in names(trF)){
#   trF[[f]] <- as.numeric(trF[[f]])
# }
# X <-  mRMR.data(data = trF)
# mm <- sort(mim(X)[, 1], decreasing = T)

###
tr <- train[, c("ID", "TARGET", selectedFeats, leftOuts)]
te <- test[, c("ID", "TARGET", selectedFeats, leftOuts)]
write_csv(tr, './RawData/trainClean.csv')
write_csv(te, './RawData/testClean.csv')

##
feats <- setdiff(names(tr), c("ID", "TARGET"))
kf <- createFolds(tr$TARGET, k = 10, list = T)
cv <- c()
iter <- c()
for(i in 1:length(kf)){
  idx <- kf[[i]]
  trS <- tr[-idx,]
  valS <- tr[idx,]
  dtrain <- xgb.DMatrix(data = data.matrix(trS[,feats]), label = trS$TARGET, missing = NA)
  dval <- xgb.DMatrix(data = data.matrix(valS[,feats]), label = valS$TARGET, missing = NA)
  param <- list(objective = 'binary:logistic',
                 max_depth = 5,
                 eta = 0.013,
                 subsample = 0.69,
                 colsample_bytree = 0.7,
                 eval_metric = 'auc')
  watchlist <- list(OOB = dval, train = dtrain)
  clf <- xgb.train(params = param,
                   data = dtrain,
                   nrounds = 3000,
                   watchlist = watchlist,
                   early.stop.round = 160,
                   print.every.n = 100)
  cv <- c(cv, clf$bestScore)
  iter <- c(iter, clf$bestInd)
}
dtrain <- xgb.DMatrix(data = Matrix(data.matrix(tr[,feats]), sparse = T), label = tr$TARGET)
watchlist <- list(train = dtrain)
clf <- xgb.train(params = param,
                 data = dtrain,
                 nrounds = 900,
                 watchlist = watchlist,
                 print.every.n = 100)
preds <- predict(clf, Matrix(data.matrix(te[, feats]), sparse = T))
sub <- data.frame(ID = te$ID, TARGET = preds)
write.csv(sub, "./Submissions/xgb_best_80_MRMR_sparse.csv",row.names = F)

##
evalList <- list()
best80 <- list(cv = cv,
               iter.optim = 847,
               meanCV = mean(cv),
               sdCV = sd(cv),
               param = param,
               folds = kf)
evalList$top80 <- best80
## select top 100

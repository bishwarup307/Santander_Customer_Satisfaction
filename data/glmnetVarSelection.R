library(caret)
library(glmnet)
library(readr)
library(data.table)
library(doParallel)
library(gtools)
library(pROC)
# setwd("/home/branden/Documents/kaggle/bnp")
setwd("F:/Kaggle/Santander/BrandenData/")
threads <- 12
ts1Trans <- data.frame(fread("./ts2Trans_v11.csv"))
load("./data_trans/cvFoldsTrainList.rda")

varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","TARGET","filter","dummy","pred0"), with=FALSE]))

# # Was only necessary for easier filtering of the validation set
# train <- ts1Trans[ts1Trans$filter==0,]
# test <- ts1Trans[ts1Trans$filter==2,]
# # Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
# train$target <- as.factor(make.names(train$target))
# # 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)


# pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","pca","center","scale"), thresh=0.999)
# pca <- preProcess(ts1Trans[filter==0,varnames, with=FALSE], method=c("zv","BoxCox","center","scale"))
# train_pca <- predict(pca, train[,4:ncol(train), with=FALSE])
# test_pca <- predict(pca, newdata=test[,4:ncol(test), with=FALSE])
# ts1Trans <- predict(pca, newdata=ts1Trans[,varnames,with=FALSE])

# rm(train, test)

# Logloss function
# LogLoss <- function(actual, predicted, eps=1e-15) {
#   predicted[predicted < eps] <- eps;
#   predicted[predicted > 1 - eps] <- 1 - eps;
#   -1/nrow(actual)*(sum(actual*log(predicted)))
# }
tr <- ts1Trans[ts1Trans$filter == 0,]
feature.names <- setdiff(names(tr), c("ID","TARGET","filter","dummy","pred0"))
gcv <- cv.glmnet(x = as.matrix(tr[, feature.names]),
                 y = tr$TARGET,
                 family = 'binomial',
                 type.measure = 'auc',
                 nfolds = 4)
coeffs <- as.matrix(coef(gcv, s = 'lambda.1se'))
chosenVars <- rownames(coeffs)[abs(coeffs[,1]) > 0][-1]
tt <- ts1Trans[, c('ID', 'TARGET', 'filter', chosenVars)]
write_csv(tt, './glmnetVarSelectAlldata.csv')


glmnetControl <- trainControl(method="cv",
                              number=4,
                              summaryFunction=twoClassSummary,
                              savePredictions=TRUE,
                              classProbs=TRUE,
                              allowParallel=TRUE)
glmnetGrid <- expand.grid(alpha=c(.09,.1,.11), lambda=c(.003,.004,0.005))

cl <- makePSOCKcluster(threads)
clusterEvalQ(cl, library(foreach))
registerDoParallel(cl)
set.seed(201601)
(tme <- Sys.time())
glmnet4 <- train(x=ts1Trans[filter==0,6:ncol(ts1Trans),with=FALSE],
                 y=as.factor(make.names(ts1Trans[ts1Trans$filter==0,TARGET])),
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="ROC")
stopCluster(cl)
Sys.time() - tme
save(glmnet4, file="./stack_models/layer1_glmnet4.rda")

cvPreds <- glmnet4$pred[glmnet4$pred$alpha==glmnet4$bestTune$alpha & glmnet4$pred$lambda==glmnet4$bestTune$lambda,c(3,5)]
cvPreds <- cvPreds[order(cvPreds$rowIndex),]
cvPreds$rowIndex <- NULL

colnames(cvPreds) <- "PredictedProb_glmnet4"
write_csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], cvPreds), "./stack_models/cvPreds/cvPreds_glmnet4.csv") 

# Test Predictions and Submission file
preds <- predict(glmnet4, newdata=ts1Trans[filter==2,6:ncol(ts1Trans),with=FALSE], type="prob")[,2]
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=samp$ID, TARGET=preds)
write_csv(submission, "./stack_models/testPreds/testPreds_glmnet4.csv")


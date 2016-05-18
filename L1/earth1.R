##################
# cv : 0.8273074 #
##################

# load required libraries
require(readr)
require(Hmisc)
require(Metrics)
require(caret)
require(doParallel)

# set working directory and seed
dir <- 'F:/Kaggle/Santander/'
setwd(dir)
eval_dir <- './evals/'
test_dir <- './tests/'
set.seed(201)

# read the data file
alldata <- read.csv('./glmnetVarSelectAlldata.csv')
fold_ids <- read.csv('./Fold5F.csv')

train <- alldata[alldata$filter == 0,]
test <- alldata[alldata$filter == 2,]
feature.names <- names(train)[!names(train) %in% c("ID", "TARGET", "filter")]


cl <- makePSOCKcluster(12)
registerDoParallel(cl)

# trCtrl <- trainControl(method = 'cv',
#                        number = 5,
#                        verboseIter = TRUE,
#                        classProbs = TRUE,
#                        summaryFunction=twoClassSummary,
#                        allowParallel = TRUE)
# 
# tuneGr <- expand.grid(nprune = c(30, 40, 50), degree = 2)
# 
# earthM <- train(x=train[, feature.names],
#                 y = as.factor(make.names(train$TARGET)),
#                 method = 'earth',
#                 trControl = trCtrl,
#                 tuneGrid = tuneGr,
#                 preProcess = c('center', 'scale'),
#                 metric= 'ROC')
# stopCluster(cl)

# best tune
# prune: 30, degree :2 

trCtrl <- trainControl(method = 'none',
                       verboseIter = TRUE,
                       classProbs = TRUE,
                       summaryFunction=twoClassSummary,
                       allowParallel = TRUE)
tunedParams <- expand.grid(nprune = 30, degree = 2)

# meta containers
evalMatrix <- data.frame(ID = numeric(), earth1_preds = numeric())
testMatrix <- data.frame(ID = test$ID)

cv <- c()
for(ii in 1:ncol(fold_ids)){
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", ii, "----------")
  cat("\n---------------------------\n")
  cname <- paste("Fold_", ii)
  idx <- fold_ids[[ii]]
  idx <- idx[!is.na(idx)]
  
  trainingSet <- train[!train$ID %in% idx,]
  validationSet <- train[train$ID %in% idx,]
  
  cat("\nnrow train: ", nrow(trainingSet))
  cat("\nnrow eval: ", nrow(validationSet), "\n")
  
  cl <- makePSOCKcluster(12)
  registerDoParallel(cl)
  
  earth_ <- train(x=trainingSet[, feature.names],
                  y = as.factor(make.names(trainingSet$TARGET)),
                  method = 'earth',
                  trControl = trCtrl,
                  tuneGrid = tunedParams,
                  #preProcess = c('center', 'scale'),
                  metric= 'ROC')
  preds <- predict(earth_, newdata = validationSet[,feature.names], type = 'prob')[, 2]
  AUC <- auc(validationSet$TARGET, preds)
  cat('\noof auc: ', AUC)
  
  cv <- c(cv, AUC)
  tmp <- data.frame(ID = validationSet$ID, earth1_preds = preds)
  evalMatrix <- rbind(evalMatrix, tmp)
  rm(tmp, trainingSet, validationSet, earth_, preds, AUC)
  gc()

}

earth_ <- train(x=train[, feature.names],
                y = as.factor(make.names(train$TARGET)),
                method = 'earth',
                trControl = trCtrl,
                tuneGrid = tunedParams,
                #preProcess = c('center', 'scale'),
                metric= 'ROC')

tpreds <- predict(earth_, newdata = test[,feature.names], type = 'prob')[, 2]
testMatrix[['earth1_preds']] <- tpreds

write_csv(evalMatrix, paste0(eval_dir, 'earth1_eval.csv'))
write_csv(testMatrix, paste0(test_dir, 'earth1_test.csv'))

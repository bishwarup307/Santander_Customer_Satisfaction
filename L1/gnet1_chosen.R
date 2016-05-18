##################
# cv : 0.8269321 #
##################

# load required libraries
require(readr)
require(Hmisc)
require(dplyr)
require(caret)
require(LiblineaR)
require(Metrics)
require(MASS)
require(glmnet)
options(dplyr.print_max = Inf)
options(dplyr.width = Inf)

# set working directory and seed
dir <- 'F:/Kaggle/Santander/'
setwd(dir)
source('./Scripts/linear_utils__.R')
set.seed(201)

# read the data file
alldata <- read.csv('./glmnetVarSelectAlldata.csv')
fold_ids <- read.csv('./Fold5F.csv')

# scale the features
useCols <- setdiff(names(alldata), c('ID', 'TARGET', 'filter'))
useDF <- alldata[, useCols]
useDF <- as.data.frame(scale(useDF))
alldata <- alldata[, !names(alldata) %in% useCols]
alldata <- cbind(alldata, useDF)

# split train & test
tr <- alldata[alldata$filter == 0,]
te <- alldata[alldata$filter == 2,]

feature.names <- names(tr)[!names(tr) %in% c("ID", "TARGET", "filter")]

# meta containers
evalMatrix <- data.frame(ID = numeric(), gnet1_preds = numeric())
testMatrix <- data.frame(ID = te$ID)

# choose best alpha and lambda
# gcv <- cv.glmnet(x = as.matrix(tr[, feature.names]),
#                  y = tr$TARGET,
#                  family = 'binomial',
#                  type.measure = 'auc',
#                  nfolds = 5)

cv <- c()
for(i in 1:ncol(fold_ids)) {
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", i, "----------")
  cat("\n---------------------------\n")
  cname <- paste("Fold_", i)
  idx <- fold_ids[[i]]
  idx <- idx[!is.na(idx)]
  
  trainingSet <- tr[!tr$ID %in% idx,]
  validationSet <- tr[tr$ID %in% idx,]
  
  cat("\nnrow train: ", nrow(trainingSet))
  cat("\nnrow eval: ", nrow(validationSet), "\n")
  
  gnet <- glmnet(x = as.matrix(trainingSet[, feature.names]),
                 y = trainingSet$TARGET,
                 family = 'binomial',
                 alpha = 0.37,
                 lambda.min.ratio = 0.013,
                 standardize = FALSE)
  
  preds <- as.numeric(predict(gnet, 
                   newx = as.matrix(validationSet[, feature.names]),
                   type = 'response',
                   s = 0.001))
  
  AUC <- auc(validationSet$TARGET, preds)
  cat('\nAUC: ', AUC)
  
  valid <- data.frame(ID = validationSet$ID, gnet1_preds = preds)
  evalMatrix <- rbind(evalMatrix, valid)
  cv <- c(cv, AUC)
  rm(trainingSet, validationSet, gnet, AUC, valid); gc()
}

gnet <- glmnet(x = as.matrix(tr[, feature.names]),
               y = tr$TARGET,
               family = 'binomial',
               alpha = 0.37,
               lambda.min.ratio = 0.013,
               standardize = FALSE)

preds <- as.numeric(predict(gnet, 
                            newx = as.matrix(te[, feature.names]),
                            type = 'response',
                            s = 0.001))
testMatrix[['gnet1_preds']] <- preds

# save to disk
write_csv(evalMatrix, './evals/gnet1_eval.csv')
write_csv(testMatrix, './tests/gnet1_test.csv')

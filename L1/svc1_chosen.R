##################
# cv : 0.8273074 #
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
evalMatrix <- data.frame(ID = numeric(), svc1_preds = numeric())
testMatrix <- data.frame(ID = te$ID)

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

  svp <- LiblineaR(as.matrix(trainingSet[, feature.names]), 
                   trainingSet$TARGET, 
                   type =  0, 
                   cost = 0.009, 
                   epsilon = 1e-7, 
                   verbose = FALSE)

  preds <- as.numeric(predict(svp, newx = as.matrix(validationSet[, feature.names]), proba = TRUE)$probabilities[, 2])
  AUC <- auc(validationSet$TARGET, preds)
  cat('\nAUC: ', AUC)

  valid <- data.frame(ID = validationSet$ID, svc1_preds = preds)
  evalMatrix <- rbind(evalMatrix, valid)
  cv <- c(cv, AUC)
  rm(trainingSet, validationSet, svp, AUC, valid); gc()
}

svp <- LiblineaR(as.matrix(tr[, feature.names]), 
                 tr$TARGET, 
                 type =  0, 
                 cost = 0.009, 
                 epsilon = 1e-7, 
                 verbose = FALSE)

tpreds <- as.numeric(predict(svp, newx = as.matrix(te[, feature.names]), proba = TRUE)$probabilities[, 2])
testMatrix[['svc1_preds']] <- tpreds

# save to disk
write_csv(evalMatrix, './evals/svc1_eval.csv')
write_csv(testMatrix, './tests/sv1_test.csv')

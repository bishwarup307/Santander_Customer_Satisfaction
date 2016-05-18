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

# minmax scaling of features
minMaxScale <- function(x) {
  
  minima <- min(x, na.rm= TRUE)
  maxima <- max(x, na.rm = TRUE)
  
  p <- (x - minima)/(maxima - minima)
  return(p)
}

# load the data files
train <- read.csv(paste0(dir, 'RawData/train.csv'))
test <- read.csv(paste0(dir, 'RawData/test.csv'))
fold_ids <- read.csv(paste0(dir, 'Fold5F.csv'))
# convert cv folds to data.frame
# ff <- read_csv('cvFolds.csv')
# fold_list<- list()
# num_folds <- length(unique(ff$foldIndex))
# for(ii in 1:num_folds){
#   ids <- train[which(ff$foldIndex == ii),]$ID
#   fold_list[[ii]] <- ids
# }
# fold_ids <- as.data.frame(fold_list)
# names(fold_ids) <- c('fold1', 'fold2', 'fold3', 'fold4', 'fold5')
# write_csv(fold_ids, paste(dir, 'fold5f.csv'))

# merge train & test
test$TARGET <- NA
alldata <- rbind(train, test)

# remove constant features
const.cols <- names(which(sapply(alldata[, -c(1, 371, 372)], function(x) length(unique(x))) == 1))
alldata <- alldata[, !names(alldata) %in% const.cols] # remove constant cols
cat('\nremoved ', length(const.cols), ' features ...')

# remove dpulicate features
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
alldata <- alldata[, !names(alldata) %in% toRemove]

# remove linearly correlated features
lin.comb <- findLinearCombos(alldata[, -c(1, 310)])
lin.comb.remove <- names(alldata[, -c(1, 310, 311)])[lin.comb$remove]
alldata <- alldata[, !names(alldata) %in% lin.comb.remove]
# 
# alldata$var3[alldata$var3 == -999999] <- -1

# use ridge regression to select only important features
tr <- alldata[which(!is.na(alldata$TARGET)),]
st <- tr[, -1]
rd <- lm.ridge(TARGET ~ ., data=st, lambda=0.5)
impo <- names(which((abs(rd$coef) > quantile(abs(rd$coef), 0.5)) == TRUE))
alldata <- alldata[, c('ID', 'TARGET', impo)]

# save(alldata, file = './cleanedLinearNN.RData')
# scale the numeric features
load('./cleanedLinearNN.RData')
bins <- grep('ind', names(alldata), value = TRUE)
binaryDF <- alldata[, bins]
alldata <- alldata[, !names(alldata) %in% bins] # drop binary cols
binaryDF[binaryDF == 0] <- -1

# unq.cnt <- sapply(alldata, function(x) length(unique(x)))
# unq.cnt <- sapply(alldata[, -c(1, 2)], function(x) length(unique(x)))
# discreteCols <- names(which(unq.cnt <= 30)) 
# discreteDF <- alldata[, discreteCols]
# for(f in names(discreteDF)){
#     discreteDF[[f]] <- as.character(discreteDF[[f]])
# }
# dmy <- dummyVars('~.', data = discreteDF, fullRank = TRUE)
# discreteDF <- data.frame(predict(dmy, discreteDF))
# denseCols <- names(which(sapply(discreteDF, sum) > 20))
# discreteDF <- discreteDF[, denseCols]
# alldata <- alldata[, !names(alldata) %in% discreteCols]

# alldata$var15 <- log1p(alldata$var15)
# alldata$var38 <- log1p(alldata$var38)
# alldata$num_var4 <- log1p(alldata$num_var4)
# alldata$num_meses_var5_ult3 <- log1p()

useCols <- setdiff(names(alldata), c('ID', 'TARGET'))

# 
# corUniv <- data.frame(Feature = character(), cor = numeric())
# for (f in useCols){
#   pearson <- cor(tr$TARGET, tr[[f]])
#   tmp <- data.frame(Feature = f, cor = abs(pearson))
#   corUniv <- rbind(corUniv, tmp)
#   rm(tmp)
# }
# corUniv <- corUniv[order(-corUniv$cor),]
# 
# corBiv <- data.frame(Feature1 = character(), Feature2 = character(), oper = integer(), cor = numeric(), lift = numeric())
# twoWay <- combn(useCols, 2, simplify = FALSE)
# for(pair in twoWay){
#     f1 <- pair[1]
#     f2 <- pair[2]
#     maxCor <- max(cor(tr$TARGET, tr[[f1]]), cor(tr$TARGET, tr[[f2]]))
#     a <- tr[[f1]] + tr[[f2]]
# }

 tr <- alldata[which(!is.na(alldata$TARGET)),]

useDF <- alldata[,useCols]
useDF$var3[useDF$var3 == -999999] <- 2
useDF$saldo_var1[useDF$saldo_var1 < 0] <- 0
useDF$delta_imp_reemb_var17_1y3[useDF$delta_imp_reemb_var17_1y3 < 0]<- 0
useDF$saldo_medio_var17_hace2[useDF$saldo_medio_var17_hace2 < 0]<- 0
useDF$saldo_medio_var33_ult1[useDF$saldo_medio_var33_ult1 < 0] <- 0

logVars <- checkLogTrafo(useDF[1:nrow(tr),], useCols, tr$TARGET)
for(f in logVars){
  useDF[[f]] <- log1p(useDF[[f]])
}
# find the two way interactions that improves correlation with
# target
# twoWay <- combn(useCols, 2, simplify = F)
# interactions <- checkCorTwoWay(useDF[1:nrow(tr),], twoWay, tr$TARGET)
# interactions <- interactions[order(-interactions$improve),]
#write_csv(interactions, './twowayLinear.csv')
inters <- read_csv('./twowayLinear.csv')
useDF$var15_num_var4 <- useDF$var15 - useDF$num_var4
useDF$var15_num_var42 <- useDF$var15 - useDF$num_var42
useDF$var15_num_meses_var5_ult3 <- useDF$var15 - useDF$num_meses_var5_ult3
useDF$var15_num_var5 <- useDF$var15 - useDF$num_var5
useDF$num_var4_num_var8_0 <- useDF$num_var4 - useDF$num_var8_0
useDF$num_var22_ult1_num_op_var41_efect_ult3 <- useDF$num_var22_ult1 * useDF$num_op_var41_efect_ult3
useDF$num_var5_0_num_var39_0 <- useDF$num_var5_0 * useDF$num_var39_0
useDF$num_var5_0_num_var41_0 <- useDF$num_var5_0 * useDF$num_var41_0
useDF$num_var4_num_var5_0 <- useDF$num_var4 - useDF$num_var5_0
useDF$num_var4_num_meses_var8_ult3 <- useDF$num_var4 - useDF$num_meses_var8_ult3
useDF$num_var5_0_num_var12 <- useDF$num_var5_0 + useDF$num_var12
useDF$imp_op_var39_comer_ult3_num_med_var22_ult3 <- useDF$imp_op_var39_comer_ult3 * useDF$num_med_var22_ult3
useDF$var15_num_meses_var13_corto_ult3 <- useDF$var15 - useDF$num_meses_var13_corto_ult3
useDF$num_var8_0_var38 <- useDF$num_var8_0 - useDF$var38
useDF$num_var5_0_num_var12_0 <- useDF$num_var5_0 + useDF$num_var12_0

ap <- useDF[1:nrow(tr),]
ap <- cbind(tr$TARGET, ap)
names(ap)[1] <- 'TARGET'
rd <- lm.ridge(tr$TARGET ~ ., data=ap, lambda=0.4)
impo <- names(which((abs(rd$coef) > quantile(abs(rd$coef), 0.5)) == TRUE))
useDF <- useDF[, impo]
# excludeCols <- c('saldo_var5', 'delta_imp_aport_var13_1y3', 'saldo_medio_var8_ult1')
# logDF <- useDF[, !names(useDF) %in% excludeCols]
# logDF <- as.data.frame(log1p(logDF)^0.6)
# useDF <- useDF[, !names(useDF) %in% names(logDF)]
# useDF <- cbind(useDF, logDF)
useDF <- data.frame(scale(useDF))

#useDF<- data.frame(sapply(useDF, minMaxScale)) # apply minmax scalar
alldata <- alldata[, !names(alldata) %in% useCols]

alldata <- cbind(alldata, binaryDF)
alldata <- cbind(alldata, useDF)

# 
# useDF <- log1p(useDF)
# alldata <- alldata[, !names(alldata) %in% useCols]
# alldata <- cbind(alldata, useDF)

# split train & test
tr <- alldata[which(!is.na(alldata$TARGET)),]
te <- alldata[which(is.na(alldata$TARGET)),]

#

gcv <- cv.glmnet(x = as.matrix(tr[, feature.names]),
                 y = tr$TARGET,
                 type.measure = 'auc',
                 family = 'binomial',
                 nfolds = 5)

coeffs <- as.matrix(coef(gcv, s = 'lambda.1se'))
selected <- rownames(coeffs)[abs(coeffs[,1]) > 0][-1]

# train and test meta feature containers
evalMatrix <- data.frame(ID = numeric(), svc1_preds = numeric())
testMatrix <- data.frame(ID = te$ID)

# features to use in the model
# feature.names <- names(tr)[!names(tr) %in% c("ID", "TARGET")]
feature.names <- chosenVars

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
  
  frm <- as.formula(paste0('TARGET ~ ', paste(feature.names, collapse = ' + ')))
  lr <- glm(frm, family = 'binomial',
            data = trainingSet)
  p <- as.numeric(predict(lr, validationSet[, feature.names], type = 'response'))
  auc(validationSet$TARGET, p)
#   
  svp <- LiblineaR(as.matrix(trainingSet[, feature.names]), trainingSet$TARGET, type =  0, cost = 3, epsilon = 1e-7, verbose = TRUE)
  preds <- as.numeric(predict(svp, newx = as.matrix(validationSet[, feature.names]), proba = TRUE)$probabilities[, 2])
  AUC <- auc(validationSet$TARGET, preds)
  cat('\nAUC: ', AUC)
  
  enet <- glmnet(x = as.matrix(trainingSet[, feature.names]),
                 y = trainingSet$TARGET,
                 alpha = 0.8,
                 family = 'binomial',
                 lambda.min.ratio = 0.03,
                 standardize = FALSE)
  
  
  preds <- as.numeric(predict(enet, newx = as.matrix(validationSet[,feature.names]), type = "response", s = 1e-5))
  auc(validationSet$TARGET, preds)
  
  valid <- data.frame(ID = validationSet$ID, svc1_preds = preds)
  evalMatrix <- rbind(evalMatrix, valid)
  gc()
}
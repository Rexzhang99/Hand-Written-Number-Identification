# # download data from http://yann.lecun.com/exdb/mnist/
# download.file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
#               "train-images-idx3-ubyte.gz")
# download.file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
#               "train-labels-idx1-ubyte.gz")
# download.file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
#               "t10k-images-idx3-ubyte.gz")
# download.file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
#               "t10k-labels-idx1-ubyte.gz")
# # gunzip the files
# R.utils::gunzip("../Data/train-images-idx3-ubyte.gz")
# R.utils::gunzip("../Data/train-labels-idx1-ubyte.gz")
# R.utils::gunzip("../Data/t10k-images-idx3-ubyte.gz")
# R.utils::gunzip("../Data/t10k-labels-idx1-ubyte.gz")

rm(list=ls())
setwd("/Users/Trent/Documents/Study/Courses/2020Spring/Data Mining II/Homework/Final Project/Code")
set.seed(1234567)



############load data##########
# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images 
train0 = load_image_file("../Data/train-images-idx3-ubyte")
test0  = load_image_file("../Data/t10k-images-idx3-ubyte")

# load labels
train0$y = as.factor(load_label_file("../Data/train-labels-idx1-ubyte"))
test0$y  = as.factor(load_label_file("../Data/t10k-labels-idx1-ubyte"))

train=sample(train0,5000)
test=sample(test0,5000)

# view test image
show_digit(train[1000, ])

# # testing classification on subset of training data
# fit = randomForest::randomForest(y ~ ., data = train[1:1000, ])
# fit$confusion
# test_pred = predict(fit, test)
# mean(test_pred == test$y)
# table(predicted = test_pred, actual = test$y)


########testing NN#######
source("NeuralNetworkC.R")

training_data <- train_test_from_df(df = train, predict_col_index = 785, train_ratio = 1)[[1]]
testing_data <- train_test_from_df(df = test, predict_col_index = 785, train_ratio = 1)[[1]]

in_n <- length(training_data[[1]][[1]])
out_n <- length(training_data[[1]][[-1]])
epochs=20
lr=0.1
wd=0
mini_batch_size=10

trained_net <- neuralnetwork(
  c(in_n,10, out_n),
  training_data=training_data,
  epochs,
  mini_batch_size,
  lr,
  C='ce',
  verbose=TRUE,
  validation_data=testing_data
)

biases <- trained_net[[1]]
weights <- trained_net[[-1]]

# Accuracy (train)
Accuracy.train=evaluate(training_data, biases, weights)
# Accuracy (test)
Accuracy.test=evaluate(testing_data, biases, weights)

library(ggplot2)
t = reshape::melt(Accuracy.train[[1]])
ggplot(t, aes(t[, 1], t[, 2], fill = value, label = round(value, 3))) + # x and y axes => Var1 and Var2
    geom_tile() + # background colours are mapped according to the value column
    geom_text() +
    scale_fill_continuous(high = "#c0e1fa", low = "#ffffff") +
    theme(legend.position = "bottom",
          panel.background = element_rect(fill = "white")) +
    scale_x_discrete(label = abbreviate) + scale_y_discrete(label = abbreviate) +
    xlab(paste("Predicted class")) + ylab("Actual class")+labs(title = paste("Accuracy.train=",round(Accuracy.train[[2]],4)))
ggsave("Predicted class vs Actual class.png", path = "../Report/figure", scale = 0.8)

t = reshape::melt(Accuracy.test[[1]])
ggplot(t, aes(t[, 1], t[, 2], fill = value, label = round(value, 3))) + # x and y axes => Var1 and Var2
  geom_tile() + # background colours are mapped according to the value column
  geom_text() +
  scale_fill_continuous(high = "#c0e1fa", low = "#ffffff") +
  theme(legend.position = "bottom",
        panel.background = element_rect(fill = "white")) +
  scale_x_discrete(label = abbreviate) + scale_y_discrete(label = abbreviate) +
  xlab(paste("Predicted class")) + ylab("Actual class")+labs(title = paste("Accuracy.test=",round(Accuracy.test[[2]],4)))
ggsave("Predicted class vs Actual class t.png", path = "../Report/figure", scale = 0.8)
















########SVM########
source("ErrorAnalysis.R")
library(kernlab)
x=as.matrix(train[,-785])
y=as.matrix(as.numeric(train$y))





svm=ksvm(x,y,type = "C-svc",scale = FALSE,cross=5)
predict.svm=predict(svm, test[, 1:784])
error_analysis(predict.svm,as.numeric(test[, 785]),'SVM')


svm=ksvm(x,y,type = "C-svc",scale = FALSE,cross=5,kernel="rbfdot")
predict.svm=predict(svm, test[, 1:784])
error_analysis(predict.svm,as.numeric(test[, 785]),'SVM')

# source("SSGSVM.R")
# cost <- function(x, y, a, b){
#   threshold <- x %*% a + b
#   predicted <- rep(0,length(y))
#   predicted[threshold < 0] <- -1
#   predicted[threshold >= 0] <- 1
#   return(sum(predicted == y) / length(y))
# }
# grad <- function(x, y, a, b, lambda) {
#   hard_margin <- y * (x %*% a + b) # define hard-margin SVM
#   if (hard_margin >= 1) {
#     # point is not within the hard-margin, so increase the margin
#     gradient <- c(b, a * lambda)
#   } 
#   else {
#     # point is within the hard-margin, so decrease the margin
#     gradient <- c(-y, (a * lambda) - (y * x))
#   }
#   return(gradient)
# }
# 
# x = x
# y = y
# init = rep(0, ncol(x))
# epochs = 50
# eta = 0.1
# cost = cost
# grad = grad
# lambda = 0.01
# param <- SVM_Regularization(x = x,
#                             y = y,
#                             init = init,
#                             epochs = epochs,
#                             eta = eta,
#                             cost = cost,
#                             grad = grad,
#                             lambda = lambda)




# a=vector(mode = "list", length = 10*10)
# for (i in 1:10) {
#   C=2^(i-6)
#   for (j in 1:10) {
#     epsilon=0.2*(j-6)
#     temp=ksvm(X,Y,type = "C-svc",kernel = "rbfdot",C=C,epsilon=epsilon,cross=2)
#     if(temp@error<lasttemp@error) best.model=temp
#     if(temp@cross<lasttemp@cross) best.cross.model=temp
#     a=c(a,temp)
#     lasttemp=temp
#   }
# }
# best.model
# best.cross.model
# mse.best.model=best.model@cross
# mse.best.cross.model=best.cross.model@cross

t=as.data.frame(rbind(c("","Ridge","SVR"),
                      c("MSE",mse.min,mse.best.cross.model)))
colnames(t) = NULL
xtable::xtable(t)

predict.svm=predict(svm, test[, 1:784])
error_analysis(predict.svm,as.numeric(test[, 785]),'SVM')


# library(GGally)
# data=train
# temp=ggpairs(data[sample(nrow(data), 0.001 * nrow(data)), ], lower = list(continuous = wrap(
#   "points", alpha = 1, size = 0.1
# )))


#new#######
library("mlr")
library('ggplot2')
class.task=makeClassifTask(data=as.big.matrix(train0[1:5000,]),target="y")
class.task
getDefaultMeasure(class.task)

n = getTaskSize(class.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)

class.lrn=makeLearner("classif.ksvm",predict.type='prob')
class.lrn
class.lrn$par.set


##default parameter####
mod=train(class.lrn,class.task, subset = train.set)
pred = predict(mod, task = class.task, subset = test.set)
head(as.data.frame(pred))

conf.matrix =calculateConfusionMatrix(pred)
conf.matrix

##manual tune#####
num_ps = makeParamSet(
  makeNumericParam("C", lower = -10, upper = 10, trafo = function(x) 10^x),
  makeNumericParam("sigma", lower = -10, upper = -7, trafo = function(x) 10^x)
)
ctrl = makeTuneControlRandom(maxit = 20L)
rdesc = makeResampleDesc("CV", iters = 2L)

res = tuneParams(class.lrn, task = class.task, resampling = rdesc,par.set = num_ps, control = ctrl)
res
res.data=generateHyperParsEffectData(res)
res.data
plotHyperParsEffect(res.data, x = "iteration", y = "mmce.test.mean",plot.type = "line")+
  theme_minimal()
plotHyperParsEffect(res.data, x = "C", y = "mmce.test.mean",plot.type = "line")+
  theme_minimal()
plotHyperParsEffect(res.data, x = "sigma", y = "mmce.test.mean",plot.type = "line")+
  theme_minimal()
plotHyperParsEffect(res.data, x = "C", y = "sigma",z= "mmce.test.mean",plot.type = "scatter", interpolate = "regr.earth", show.experiments = TRUE)+
  theme_minimal()















#new#######
library("mlr")
library('ggplot2')
class.task=makeClassifTask(data=as.big.matrix(train0[1:5000,]),target="y")
class.task
getDefaultMeasure(class.task)

n = getTaskSize(class.task)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)

class.lrn=makeLearner("classif.nnTrain",predict.type='prob')
class.lrn
class.lrn$par.set


##default parameter####
mod=train(class.lrn,class.task, subset = train.set)
pred = predict(mod, task = class.task, subset = test.set)
head(as.data.frame(pred))

conf.matrix =calculateConfusionMatrix(pred)
conf.matrix

##manual tune#####
num_ps = makeParamSet(
  makeNumericParam("C", lower = -10, upper = 10, trafo = function(x) 10^x),
  makeNumericParam("sigma", lower = -10, upper = -7, trafo = function(x) 10^x)
)
ctrl = makeTuneControlRandom(maxit = 20L)
rdesc = makeResampleDesc("CV", iters = 2L)

res = tuneParams(class.lrn, task = class.task, resampling = rdesc,par.set = num_ps, control = ctrl)
res
res.data=generateHyperParsEffectData(res)
res.data
plotHyperParsEffect(res.data, x = "iteration", y = "mmce.test.mean",plot.type = "line")+
  theme_minimal()
plotHyperParsEffect(res.data, x = "C", y = "mmce.test.mean",plot.type = "line")+
  theme_minimal()
plotHyperParsEffect(res.data, x = "sigma", y = "mmce.test.mean",plot.type = "line")+
  theme_minimal()
plotHyperParsEffect(res.data, x = "C", y = "sigma",z= "mmce.test.mean",plot.type = "scatter", interpolate = "regr.earth", show.experiments = TRUE)+
  theme_minimal()








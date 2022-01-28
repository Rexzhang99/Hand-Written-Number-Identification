library(glmnet)
library(ggplot2)
library(Metrics)
library(FNN)
library(caret)
library(gridExtra)
rm(list = ls())
set.seed(5665)
setwd(
  "Documents/Study/Courses/2019Fall/Data Mining Methodology I/Homework/Final project"
)

error_analysis <- function(pred,actual, plot_name) {
  # pred = predict(model, DT.test)
  error = ifelse(pred == actual, 1, 0)
  table.error = table(data.frame(pred, actual))
  table.error.rate = table.error 
  errorrate = sum(error) / length(error)
  t = reshape2::melt(table.error.rate)
  ep = ggplot(t, aes(t[, 1], t[, 2], fill = value, label = round(value, 3))) + # x and y axes => Var1 and Var2
    geom_tile() + # background colours are mapped according to the value column
    geom_text() +
    scale_fill_continuous(high = "#c0e1fa", low = "#ffffff") +
    theme(legend.position = "bottom",
          panel.background = element_rect(fill = "white")) +
    scale_x_discrete(label = abbreviate) + scale_y_discrete(label = abbreviate) +
    xlab(paste("Predicted class by", plot_name)) + ylab("Actual class")
  # +ggtitle(paste("Overall Accuracy=", round(model$results$Accuracy, 4)))
  # cvep = ggplot(model$resample, mapping = aes(Resample, Accuracy)) +
  #   geom_point() +
  #   geom_hline(yintercept = mean(model$resample$Accuracy))
  summa = list()
  summa[["error"]] = error
  summa[["table.error"]] = table.error
  summa[["table.error.rate"]] = table.error.rate
  summa[["errorrate"]] = errorrate
  ggsave(
    file = paste("error", gsub(" ", "", plot_name), '.png', sep = ''),
    path = "Thesis/figure",
    # grid.arrange(cvep, ep, ncol = 2),
    ep,
    width = 10,
    height = 5
  )
  return(summa)
}


# r.dt=data.table::fread("/Users/Trent/Downloads/track_features/tf_000000000000.csv", header = T, sep = ',')
# select_index=sample(nrow(r.dt),100000)
# dt = as.data.frame(r.dt[select_index,])
# rownames(dt.train) <- NULL

# dt=read.csv("data/track_features/tf_mini.csv")
# dt=dt[,2:30]
# index.pop=which(dt[,3]>98)
# dt=dt[index.pop,]
# index.year=which(dt[,2]==2018)
# dt=dt[index.year,c(1,3:29)]

t = read.csv("data/SpotifyAudioFeaturesApril2019.csv")
data.nnum = t[, 1:3]
data = t[, 4:17]

train = sample(nrow(data), 0.7 * nrow(data))
data.train = data[train,]
data.test = data[-train,]
############################
ggplot(data, aes(popularity)) + geom_histogram()

library(GGally)
temp=ggpairs(data[sample(nrow(data), 0.1 * nrow(data)), ], lower = list(continuous = wrap(
  "points", alpha = 0.3, size = 0.1
)))
ggsave(file = 'ggpairs.png',
       path = "Thesis/figure",
       temp,
       width = 20,
       height = 10
)

############################regression of popular type
dt.train = data.train[which(data.train$popularity>79),]
dt.test = data.test[which(data.test$popularity>79),]

lm1 = lm(popularity ~ ., data = dt.train)
summary(lm1)
x = dt.train$popularity
y = lm1[["fitted.values"]]
predict_and_observe_plot_data = data.frame(x, y)
p = ggplot(data = predict_and_observe_plot_data, aes(x, y)) +
  geom_point() +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "red",
    linetype = "dashed",
    size = 1
  ) +
  xlab("Observed Critical Temperature (K)") +
  ylab("Predicted Critical Temperature (K)")
ggsave(
  file = 'predict_and_observe_plot.eps',
  path = "Thesis/figure",
  gridExtra::arrangeGrob(p),
  width = 10,
  height = 5
)
rmse.lm.train = rmse(dt.train$popularity, lm1[["fitted.values"]])
error.lm.train = dt.test$popularity - lm1[["fitted.values"]]


pred.lm = predict(lm1, dt.test[, -14])
x = dt.test$popularity
y = pred.lm
predict_and_observe_plot_data = data.frame(x, y)
p = ggplot(data = predict_and_observe_plot_data, aes(x, y)) +
  geom_point() +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "red",
    linetype = "dashed",
    size = 1
  ) +
  xlab("Observed Critical Temperature (K)") +
  ylab("Predicted Critical Temperature (K)")
ggsave(
  file = 'testerror.png',
  path = "Thesis/figure",
  gridExtra::arrangeGrob(p),
  width = 10,
  height = 5
)
rmse.lm.test = rmse(dt.test$popularity, pred.lm)
error.lm.test = dt.test$popularity - pred.lm

index.outliner = which(abs(error.lm) > 3 * rmse.lm)


x = as.matrix(dt.train[,-14])
y = as.matrix(dt.train$popularity)
x.t = as.matrix(dt.test[,-14])
y.t = as.matrix(dt.test$popularity)

cvfit.ridge = cv.glmnet(x, y, alpha = 0)
png('Thesis/figure/cvfitridge.png',width = 600, height = 300)
plot(cvfit.ridge)
dev.off()
xtable(as.array(glmnet(x, y, alpha = 0, lambda = cvfit.ridge$lambda.min)$beta))

cvfit.lasso = cv.glmnet(x, y, alpha = 1)
png('Thesis/figure/cvfitlasso.png',width = 600, height = 300)
plot(cvfit.lasso)
dev.off()
xtable(as.array(glmnet(x, y, alpha = 1, lambda = cvfit.lasso$lambda.min)$beta))


png('Thesis/figure/cvfitpla_net.png')
par(mfrow=c(3,3))
pla_net=function(k){
  cvfit.net = cv.glmnet(x, y, alpha=k)
  plot(cvfit.net, sub=paste("Alpha=",k))
}
for (k in 0.1*1:9) {
  pla_net(k)
}
dev.off()
xtable(as.array(glmnet(x, y, alpha = 0.5, lambda = 0.5)$beta))

f_knn_tst_rmse = function(k) {
  rmse(y.t, knn.reg(x, x.t, y, k)$pred)
}
f_knn_trn_rmse = function(k) {
  rmse(y, knn.reg(x, x, y, k)$pred)
}
# k = c(1, 5, 10, 25, 50, 250)
k = c(1, 3:10)
knn_tst_rmse = sapply(k, f_knn_tst_rmse)
knn_trn_rmse = sapply(k, f_knn_trn_rmse)
# determine "best" k
best_k = k[which.min(knn_tst_rmse)]
knn_results = data.frame(k,
                         round(knn_trn_rmse, 2),
                         round(knn_tst_rmse, 2))
colnames(knn_results) = c("k", "Train RMSE", "Test RMSE")

pred.best.knn = knn.reg(x, x.t, y, 25)$pred
error.best.knn = y.t - pred.best.knn
plot(error.best.knn)





#######################classification
DT.train = data.train
DT.test = data.test
DT.train$popularity = as.factor(ifelse(
  DT.train$popularity < 10,
  "c.unpopular",
  ifelse(
    DT.train$popularity < 40,
    "b.normal",
    ifelse(DT.train$popularity < 70, "b.normal", "a.popular")
  )
))
DT.test$popularity = as.factor(ifelse(
  DT.test$popularity < 10,
  "c.unpopular",
  ifelse(
    DT.test$popularity < 40,
    "b.normal",
    ifelse(DT.test$popularity < 70, "b.normal", "a.popular")
  )
))
#c.unpopular
table(DT.test$popularity)

plot(DT.test$popularity)
#############Logistic Regression
ctrl = trainControl(method = "cv", 3)
model.multinom = train(
  popularity ~ .,
  data = DT.train,
  method = "multinom",
  trControl = ctrl,
  preProcess = c("center", "scale")
)
pred.multinom = predict(model.multinom, DT.test, type = "prob")
error.multinom = error_analysis(predict(model.multinom, DT.test),DT.test[, 14], plot_name = "Multinomial Logistic Regression")
error.multinom
#############LDA
ctrl = trainControl(method = "cv", 10)
model.lda = train(
  popularity ~ .,
  data = DT.train,
  method = "lda",
  trControl = ctrl,
  preProcess = c("center", "scale")
)
pred.lda = predict(model.lda, DT.test, type = "prob")
error.lda = error_analysis(predict(model.lda, DT.test),DT.test[, 14], plot_name = "Linear Discriminative Analysis")
error.lda
#############QDA
ctrl = trainControl(method = "cv", 10)
model.qda = train(
  popularity ~ .,
  data = DT.train,
  method = "qda",
  trControl = ctrl,
  preProcess = c("center", "scale")
)
pred.qda = predict(model.qda, DT.test, type = "prob")
error.qda = error_analysis(predict(model.qda, DT.test),DT.test[, 14], plot_name = "Quadratic Discriminative Analysis")
error.qda
#############KNN
pred.knn = knn(DT.train[, 1:13], DT.test[, 1:13], DT.train[, 14], k = 2, prob = TRUE)
error.knn = error_analysis(as.vector(pred.knn),DT.test[, 14], plot_name = "K-Nearest Neighbors")
error.knn$table.error

f_knn = function(k) {
  error.knn =error_analysis(as.vector(knn(DT.train[, 1:13], DT.test[, 1:13], DT.train[, 14], k, prob = TRUE)),DT.test[, 14], plot_name = "K-Nearest Neighbors")
  print(error.knn$table.error)
}
k = c(1:10)
sapply(k, f_knn)

View(t[row.names(DT.test[which(as.vector(pred.knn) != DT.test$popularity & DT.test$popularity == "c.unpopular"), ]),c(1,3,17)])
#############Dicision tree
model.tree = rpart::rpart(popularity ~ ., data = DT.train)
pred.tree = predict(model.tree, DT.test[, 1:13], type = "class")
error.tree = error_analysis(as.vector(pred.tree),DT.test[, 14], plot_name = "Dicision Tree")
error.tree
#############Random forest
# model.forest= randomForest::randomForest(popularity ~., data=DT.train)
# pred.forest= predict(model.forest, DT.test[,1:13], type = "class")
# error.forest=error_analysis(as.vector(pred.tree),plot_name="Random Forest")



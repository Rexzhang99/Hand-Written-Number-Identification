accuracies = read.csv('Result/nist_tests.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('hidden_nodes', 'learning_rate', 'bias', 'epoch', 'train corrects rate', 'train wrongs rate', 'test corrects rate', 'test wrongs rate')
accuracies = accuracies[, -c(6, 8)]
accuracies = reshape2::melt(accuracies, id.vars = c('hidden_nodes', 'learning_rate', 'bias', 'epoch'))
accuracies[c('hidden_nodes', 'learning_rate', 'bias')]=lapply(accuracies[c('hidden_nodes', 'learning_rate', 'bias')], factor)

library(ggplot2)
library(ggrepel)
dt=accuracies
for (x in c('bias')) {
  ggplot(dt,aes(dt[[x]],value,fill=variable))+
    geom_boxplot() +
    xlab(x)+ 
    stat_summary(fun.y=max, geom="line", aes(group=variable,color=variable))+
    stat_summary(fun.y=max, geom="point", aes(group=variable,color=variable))+
    stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
    theme_bw()+ theme(legend.position="bottom")+ylab('Accuracy')
    # title('hidden_nodes=20, 50, 100, 120, 150\n learning_rate=0.01, 0.05, 0.1, 0.2\n bias=None, 0.5  epoch=1,2,3')
  ggsave(paste('1-Hidden Layer Neural Network Accuracy rate versus ',x,'.png',sep = ''), path ='../Report/figure', scale = 0.6)
}




# accuracies = read.csv('Result/nist_tests_Multiple.csv', sep = ' ', header = FALSE)
# colnames(accuracies) = c('layer_one_nodes','layer_two_nodes','train corrects rate','train wrongs rate','test corrects rate','test wrongs rate')
# accuracies = accuracies[, -c(4, 6)]
# accuracies = reshape2::melt(accuracies, id.vars = c('layer_one_nodes', 'layer_two_nodes'))
# 
# ggplot(accuracies, aes(layer_one_nodes, layer_two_nodes,color=variable, shape = variable,size=value,label=round(value,4))) +
#   geom_point(alpha=0.6)+
#   ggrepel::geom_text_repel(size = 3,alpha=0.6,box.padding=3,segment.alpha=0.6)+
#   ylab('Accuracy')+
#   theme_bw()+ theme(legend.position="bottom")
# ggsave(paste('2-Hidden Layer Neural Network Accuracy rate, layer_one_nodes versus layer_two_nodes.png',sep = ''), path ='../Report/figure', scale = 0.6)











accuracies = read.csv('Result/nist_tests_keras.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('hidden_nodes','activation','batch_size','loss','learning_rate','epoch','train_accuracy','test_accuracy','train_loss','test_loss')
accuracies = reshape2::melt(accuracies,measure.vars = c('train_accuracy','test_accuracy','train_loss','test_loss'), id.vars = c('hidden_nodes','activation','batch_size','loss','learning_rate','epoch'))
accuracies=tidyr::unite(accuracies, col, 'hidden_nodes':'learning_rate', sep = "_", remove = FALSE, na.rm = FALSE)
accuracies[c('hidden_nodes','activation','batch_size','loss','learning_rate')]=lapply(accuracies[c('hidden_nodes','activation','batch_size','loss','learning_rate')], factor)
accuracies=subset(accuracies,learning_rate!=0.1)

dt=subset(accuracies, variable %in% c('test_accuracy')&activation=='softmax')
ggplot(dt, aes(epoch, value, group=col,color=hidden_nodes)) +
  geom_line()+ 
  xlim(0,9)+
  ggrepel::geom_label_repel(data=subset(dt, value<0.91&epoch==9),aes(epoch, value,label=col),color='black',force = 10,alpha=0.7,xlim=c(0, 8),size=2)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")
ggsave(paste('1-layer Neural Network Test Accuracy By Epoch.png',sep = ''), path ='../Report/figure', scale = 0.6)

dt=subset(accuracies, variable %in% c('test_accuracy')&activation=='relu')
ggplot(dt, aes(epoch, value, group=col,color=hidden_nodes)) +
  geom_line()+ 
  xlim(0,9)+
  ggrepel::geom_label_repel(data=subset(dt, value<0.91&epoch==9),aes(epoch, value,label=col),color='black',force = 10,alpha=0.7,xlim=c(0, 8),size=2)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")
ggsave(paste('91-layer Neural Network Test Accuracy By Epoch.png',sep = ''), path ='../Report/figure', scale = 0.6)


dt=subset(accuracies, variable %in% c('train_accuracy','test_accuracy')&epoch==9&activation=='softmax')
for (x in c('hidden_nodes','batch_size','loss','learning_rate')) {
  ggplot(dt,aes(dt[[x]],value,fill=variable))+
    geom_boxplot(outlier.shape = NA) +
    xlab(x)+ 
    stat_summary(fun.y=max, geom="line", aes(group=variable,color=variable))+
    stat_summary(fun.y=max, geom="point", aes(group=variable,color=variable))+
    stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
    ylim(0.9,1)+
    theme_bw()+ theme(legend.position="bottom")+ylab('Accuracy')
  ggsave(paste('1-Hidden Layer Neural Network Accuracy rate versus ',x,'.png',sep = ''), path ='../Report/figure', scale = 0.6)
}

dt=subset(accuracies, variable %in% c('train_accuracy','test_accuracy')&epoch==9&activation=='relu')
for (x in c('hidden_nodes','batch_size','loss','learning_rate')) {
  ggplot(dt,aes(dt[[x]],value,fill=variable))+
    geom_boxplot(outlier.shape = NA) +
    xlab(x)+ 
    stat_summary(fun.y=max, geom="line", aes(group=variable,color=variable))+
    stat_summary(fun.y=max, geom="point", aes(group=variable,color=variable))+
    stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
    ylim(0.9,1)+
    theme_bw()+ theme(legend.position="bottom")+ylab('Accuracy')
  ggsave(paste('91-Hidden Layer Neural Network Accuracy rate versus ',x,'.png',sep = ''), path ='../Report/figure', scale = 0.6)
}

#########Acti Comparison######

dt=subset(accuracies, variable %in% c('train_accuracy','test_accuracy')&epoch==9)
x='activation'
ggplot(dt,aes(dt[[x]],value,fill=variable))+
  geom_boxplot(outlier.shape = NA) +
  xlab(x)+ 
  stat_summary(fun.y=max, geom="line", aes(group=variable,color=variable))+
  stat_summary(fun.y=max, geom="point", aes(group=variable,color=variable))+
  stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
  ylim(0.9,1)+
  theme_bw()+ theme(legend.position="bottom")+ylab('Accuracy')
ggsave(paste('1-Hidden Layer Neural Network Accuracy rate versus ',x,'.png',sep = ''), path ='../Report/figure', scale = 0.6)

dt=subset(accuracies, variable %in% c('test_accuracy'))
ggplot(dt, aes(epoch, value, group=col,color=activation)) +
  geom_line(alpha=0.4)+ 
  xlim(0,9)+
  # ggrepel::geom_label_repel(data=subset(dt, value<0.91&epoch==9),aes(epoch, value,label=col),color='black',force = 10,alpha=0.7,xlim=c(0, 8),size=3)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")
ggsave(paste('1-Hidden layer Neural Network Accuracy rate By Epoch.png',sep = ''), path ='../Report/figure', scale = 0.6)






























accuracies = read.csv('Result/nist_tests_keras_2.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('hidden_nodes1','hidden_nodes2','activation','batch_size','loss','learning_rate','epoch','train_accuracy','test_accuracy','train_loss','test_loss')
accuracies = reshape2::melt(accuracies,measure.vars = c('train_accuracy','test_accuracy','train_loss','test_loss'), id.vars = c('hidden_nodes1','hidden_nodes2','activation','batch_size','loss','learning_rate','epoch'))
accuracies=tidyr::unite(accuracies, col, 'hidden_nodes1','hidden_nodes2':'learning_rate', sep = "_", remove = FALSE, na.rm = FALSE)
accuracies[c('hidden_nodes1','hidden_nodes2','activation','batch_size','loss','learning_rate')]=lapply(accuracies[c('hidden_nodes1','hidden_nodes2','activation','batch_size','loss','learning_rate')], factor)
accuracies=subset(accuracies,learning_rate!=0.1)

# library(plotly)
# plot_ly(x=temp, y=pressure, z=dtime, type="scatter3d", mode="markers", color=temp)
library(gridExtra)

dt=subset(accuracies, variable %in% c('test_accuracy')&activation=='relu')
p <- list()
for(i in 1:6){
  values = c(0.1,0.1,0.1,0.1,0.1,0.1)
  values[i] = 1
  p[[i]]=ggplot(dt, aes(epoch, value, group=col,color=hidden_nodes1,alpha=hidden_nodes1)) +
    geom_line()+ 
    xlim(0,9)+
    # stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
    ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")+ylim(0.85,1)+scale_alpha_manual(values = values)
}
ggsave(paste('2-layer Neural Network Test Accuracy By Epoch.png',sep = ''), plot=ggarrange(plotlist=p,ncol=6),path ='../Report/figure', scale = 2)

p <- list()
for(i in 1:6){
  values = c(0.1,0.1,0.1,0.1,0.1,0.1)
  values[i] = 1
  p[[i]]=ggplot(dt, aes(epoch, value, group=col,color=hidden_nodes2,alpha=hidden_nodes2)) +
  geom_line()+ 
  xlim(0,9)+
  # stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")+ylim(0.85,1)+scale_alpha_manual(values = values)
}
ggsave(paste('2-layer Neural Network Test Accuracy By Epoch2.png',sep = ''), plot=ggarrange(plotlist=p,ncol=6),path ='../Report/figure', scale = 2)


dt=subset(accuracies, variable %in% c('train_accuracy','test_accuracy')&epoch==9&activation=='relu')
for (x in c('hidden_nodes1','hidden_nodes2','batch_size','loss','learning_rate')) {
  ggplot(dt,aes(dt[[x]],value,fill=variable))+
    geom_boxplot(outlier.shape = NA) +
    xlab(x)+ 
    stat_summary(fun.y=max, geom="line", aes(group=variable,color=variable))+
    stat_summary(fun.y=max, geom="point", aes(group=variable,color=variable))+
    stat_summary(geom="label_repel", fun.y=max,aes(group=variable,label=sprintf("%1.4f", ..y..),alpha=0.8), size=3.5)+
    ylim(0.9,1)+
    theme_bw()+ theme(legend.position="bottom")+ylab('Accuracy')
  ggsave(paste('2-Hidden Layer Neural Network Accuracy rate versus ',x,'.png',sep = ''), path ='../Report/figure', scale = 0.6)
}



















#########SVM##########
accuracies = read.csv('Result/nist_tests_SVM.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('alpha','l1_ratio','train_accuracy','test_accuracy')
accuracies = reshape2::melt(accuracies, id.vars = c('alpha', 'l1_ratio'))


ggplot(accuracies, aes(alpha, l1_ratio, color = variable, shape = variable,size=value,label=round(value,4))) +
  geom_point(alpha=0.6)+
  ggrepel::geom_text_repel(size = 3,alpha=0.6)+
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  theme_bw()+ theme(legend.position="bottom")
ggsave(paste('SGD SVM Accuracy rate, alpha versus l1_ratio.png',sep = ''), path ='../Report/figure', scale = 0.6)




accuracies = read.csv('Result/nist_tests_SVM_rbf.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('gamma','train_accuracy','test_accuracy','time')
accuracies = reshape2::melt(accuracies, id.vars = c('gamma','time'))


ggplot(accuracies, aes(gamma, value, color = variable,label=round(value,4))) +
  geom_line(alpha=0.6)+
  geom_label(size = 3,alpha=0.6)+
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  theme_bw()+ theme(legend.position="bottom")
ggsave(paste('RBF SVM Accuracy versus gamma.png',sep = ''), path ='../Report/figure', scale = 0.6)




accuracies = read.csv('Result/nist_tests_SVM_poly.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('degree','gamma','train_accuracy','test_accuracy','time')
accuracies = reshape2::melt(accuracies, id.vars = c('degree','gamma','time'))


ggplot(accuracies, aes(gamma, degree, color = variable, shape = variable,size=value,label=round(value,4))) +
  geom_point(alpha=0.6)+
  ggrepel::geom_text_repel(size = 3,alpha=0.6)+
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  theme_bw()+ theme(legend.position="bottom")
ggsave(paste('Polynomial SVM Accuracy degree versus gamma.png',sep = ''), path ='../Report/figure', scale = 0.6)



















#######boost#######
accuracies = read.csv('Result/nist_tests_ada.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('n_estimators','learning_rate','Iteration','train_accuracy','test_accuracy','time')
accuracies = reshape2::melt(accuracies, id.vars = c('n_estimators','learning_rate','Iteration','time'))

dt=subset(accuracies,n_estimators==300,c('learning_rate','Iteration','value','variable'))
dt$learning_rate=factor(dt$learning_rate)
p=ggplot(dt, aes(Iteration,value , alpha= variable,shape = variable,color=learning_rate) )+
  geom_line()+
  geom_point(size=1)+
  ggrepel::geom_label_repel(data=subset(dt, value<0.91&Iteration%in%c(50,150,299)),aes(Iteration, value,label=round(value,4)),color='black',size=3)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")+ scale_shape( solid = FALSE)
ggsave(paste('Ada Boosting Accuracy By Iteration.png',sep = ''),p, path ='../Report/figure', scale = 0.6)

accuracies = read.csv('Result/nist_tests_ada_final.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('n_estimators','learning_rate','Iteration','train_accuracy','test_accuracy','time')
accuracies = reshape2::melt(accuracies, id.vars = c('n_estimators','learning_rate','Iteration','time'))

dt=subset(accuracies,n_estimators==300,c('learning_rate','Iteration','value','variable'))
dt$learning_rate=factor(dt$learning_rate)
p=ggplot(dt, aes(Iteration,value ,shape = variable,color=variable) )+
  geom_line()+
  geom_point(size=1)+
  ggrepel::geom_label_repel(data=subset(dt, Iteration%in%c(50,150,299)),aes(Iteration, value,label=round(value,4)),color='black',size=3)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")+ scale_shape( solid = FALSE)
ggsave(paste('Final Ada Boosting Accuracy By Iteration.png',sep = ''),p, path ='../Report/figure', scale = 0.6)




accuracies = read.csv('Result/nist_tests_gra.csv', sep = ' ', header = FALSE)
colnames(accuracies) = c('n_estimators','learning_rate','tree_depth','Iteration','train_accuracy','test_accuracy','time')
accuracies = reshape2::melt(accuracies, id.vars = c('n_estimators','learning_rate','Iteration','tree_depth','time'))

dt=subset(accuracies,n_estimators==100,c('tree_depth','learning_rate','Iteration','value','variable'))
dt$learning_rate=factor(dt$learning_rate)
dt$tree_depth=factor(dt$tree_depth)
p=ggplot(dt, aes(Iteration,value , alpha= variable, shape = tree_depth,color=learning_rate:tree_depth) )+
  geom_line()+
  geom_point(size=1)+
  ggrepel::geom_label_repel(data=subset(dt,Iteration%in%c(10,50,99)),aes(Iteration, value,label=round(value,4)),size=3)+
  # ggrepel::geom_text_repel(size = 3,alpha=0.6)+
  ylab('Accuracy')+theme_bw()+ theme(legend.position="bottom")+ scale_shape( solid = FALSE)
ggsave(paste('Gradient Boosting Accuracy By Iteration.png',sep = ''), p,path ='../Report/figure', scale = 1.2)






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
    xlab(paste("Predicted class by", plot_name)) + ylab("Actual class")+labs(title = paste('Accurancy =',round(errorrate,4)))
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
    path = "../Report/figure",
    # grid.arrange(cvep, ep, ncol = 2),
    ep,
    width = 10,
    height = 5
  )
  return(summa)
}
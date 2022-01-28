SVM_Regularization <-
  function(x, y, init, epochs, eta, cost, grad, lambda) {
    iters <- epochs * length(y)
    param <-
      bigmemory::big.matrix(nrow = iters + 1, ncol = length(init) + 2)
    options(bigmemory.allow.dimnames=TRUE)
    colnames(param) <- c("Intercept", colnames(x), "Loss")
    param[1, ] <- c(c(0, init), cost(x, y, init, 0))
    for (i in 1:iters) {
      #1 random sample for gradient
      set.seed(i)
      sample_used <- sample.int(length(y), 1)
      #parameter
      param[i + 1, 1:(length(init) + 1)] <-
        as.numeric(param[i, 1:(length(init) + 1)]) - eta * grad(x[sample_used, ], y[sample_used], as.numeric(param[i, 2:(length(init) + 1)]), param[i, 1], lambda)
      #loss
      param[i + 1, length(init) + 2] <-
        cost(x, y, as.numeric(param[i + 1, 2:(length(init) + 1)]), param[i + 1, 1])

      cat("Cost: ", sprintf("%10.07f", param[i+1, ncol(param)]), "\n", sep = "")
    }
    
    cat("Final cost: ", sprintf("%10.07f", param[nrow(param), ncol(param)]), "\n", sep = "")
    cat("Parameters:", as.numeric(param[nrow(param), 1:(length(init) + 1)]), sep = " ")
    
    param <- cbind(Iteration = 0:(nrow(param) - 1), param)
    return(param)
  }



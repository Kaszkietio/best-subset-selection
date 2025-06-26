install.packages("sparsenet")
library(sparsenet)
library(glmnet)

data_dir_list <- dir("results/datasets")
snr_list_2 <- c(3,7,10)


nonzero_beta <- function(fit){
  nonzero <- 0
  nonzero_max <- 0
  max_beta <- NULL
  for (gamma_idx in length(fit$gamma)){
    for (lambda_idx in length(fit$lambda)){
      beta <- coef(fit)[[gamma_idx]][, lambda_idx]
      nonzero <- nonzero + sum(beta != 0)
      
      if(sum(beta != 0) > nonzero_max){
        max_beta <- beta
        nonzero_max <- sum(beta != 0)
      } 
      
    }
  }
  max_beta
}

nonzero_list <- c(NULL)
mean_error_list <- c(NULL)
dataset_list  <- c(NULL)
snr_list  <- c(NULL)
method_list <- c(NULL)


for (dataset in data_dir_list){
  for (snr in snr_list_2){
    sprintf("datasets=%s snr=%s", dataset, snr)
    
    x <- sprintf("results/datasets/%s/X_snr=%s.csv", dataset, snr)
    y <- sprintf("results/datasets/%s/y_snr=%s.csv", dataset, snr)
    true_beta <- sprintf("results/datasets/%s/beta_snr=%s.csv", dataset, snr)
    
    x <- read.csv2(x, sep=',')
    x[] <- lapply(x, as.numeric)
    y <- read.csv2(y)
    true_beta <- read.csv2(true_beta)
    
    # sparsenet
    fit <- sparsenet(data.matrix(x), data.matrix(y), ngamma=2)
    beta <- coef(fit)[[1]][, 1]
    true_beta <- as.double(unlist(as.vector(true_beta)))
    
    nonzero_cols <- nonzero_beta(fit)
    nonzero <- sum(nonzero_cols != 0)
  
    intercept <- beta[1]
    beta_min <- beta[-1] 
    y_pred <- data.matrix(x) %*% data.matrix(beta_min) + intercept
    y_calc <- data.matrix(x) %*% data.matrix(true_beta)
    error <- sum(((y_pred-y_calc)^2/y_calc^2)^(1/2))
    
    nonzero_list <- c(nonzero_list, nonzero)
    mean_error_list <- c(mean_error_list, error)
    dataset_list  <- c(dataset_list, dataset)
    snr_list  <- c(snr_list, snr)
    method_list <- c(method_list, 'sparsenet')
    
    # lasso
    fit <- glmnet(data.matrix(x), data.matrix(y), alpha=1, intercept=FALSE)
    beta <- coef(fit, s=0.1)
    nonzero <- sum(beta != 0)
    
    y_pred <- data.matrix(x) %*% data.matrix(beta)[-1]
    y_calc <- data.matrix(x) %*% data.matrix(true_beta)
    error <- sum(((y_pred-y_calc)^2/y_calc^2)^(1/2))
    
    nonzero_list <- c(nonzero_list, nonzero)
    mean_error_list <- c(mean_error_list, error)
    dataset_list  <- c(dataset_list, dataset)
    snr_list  <- c(snr_list, snr)
    method_list <- c(method_list, 'lasso')
  }
}

df <- data.frame(
  nonzero=nonzero_list, error=mean_error_list, dataset=dataset_list,
  snr=snr_list, method=method_list
)
write.csv2(df)


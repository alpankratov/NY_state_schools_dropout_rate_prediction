library(tidyverse)
library(ggplot2)
library(ggthemes)
library(GGally)
library(glmnet)
library(ggcorrplot)
library(gridExtra)
library(caret)
library(e1071)
library(glue)
library(knitr)
library(kableExtra)


train <- read_csv('data/processed/train.csv')
validation <- read_csv('data/processed/validation.csv')
x_train <- select(train, -c("ind", 'aggregation_code','county_name', 'dropout_pct'))
x_validation <- select(validation, -c("ind", 'aggregation_code','county_name', 'dropout_pct'))

# Extracting response variable
y_train <- train$dropout_pct
y_validation <- validation$dropout_pct

# Use glmnet for all models - preparation of training and validation predictors
# and response variables.
x_train <- model.matrix(~.-1,data=x_train)
x_validation <- model.matrix(~.-1,data=x_validation)

# BUILDING THE MODELS - LASSO -------------------------------
# Cross-validation of lasso regression to get the optimal labda value
set.seed(123)
cv.lasso <- cv.glmnet(x_train, y_train, type.measure = 'mae', alpha = 1, family = binomial(link='logit'))
plot(cv.lasso)

log(cv.lasso$lambda.min) # -11.61754
log(cv.lasso$lambda.1se) # -6.965852
cv.lasso$lambda.1se
log(9.44e-4)
log(4.46e-2)
# fitting lasso regression with optimal lambda value (1 SE from minimal binomial deviance)
set.seed(123)
fit.lasso.min.lambda <- glmnet(x_train, y_train, alpha = 1,
                               lambda = cv.lasso$lambda.1se, family = binomial(link='logit'))
summary(fit.lasso.min.lambda)
print(fit.lasso.min.lambda)
print(cv.lasso)

coefs.lasso <- as_tibble(cbind(colnames(x_train), as_tibble(as.matrix(fit.lasso.min.lambda$beta))))
colnames(coefs.lasso) <- c('variable', 'coefficient')
coefs.lasso.nonzero <- coefs.lasso %>% filter(coefficient != 0)
write_csv(coefs.lasso, 'data/processed/lasso_betas.csv')

# computing predicted values using validation dataset (as classess and probabilities of falling into classes)
predictions.probs.lasso <- predict(fit.lasso.min.lambda, newx = x_validation, type = "response") %>% as.vector()
write_csv(as_tibble(predictions.probs.lasso), 'data/processed/lasso_yhat_val.csv')

data.frame(
  MAE = MAE(predictions.probs.lasso, y_validation),
  Rsquare.r = R2(predictions.probs.lasso, y_validation))


# BUILDING THE MODELS - ELASTIC NETS -------------------------------
# using caret package to get optimal alpha and lambda

# DEFINE CUSTOM ERROR FUNCTION
mae_loss <- function(data, lev = NULL, model = NULL) {
    mae_loss      <- mean(abs(data$obs - data$pred))
    names(mae_loss) <- "MAE"
    mae_loss }

train_net <- train %>% select(-c("ind", 'aggregation_code','county_name')) %>%
  relocate(dropout_pct, .after = last_col())

set.seed(123)
model.net <- train(
  dropout_pct ~ .-1, data = train_net, method = 'glmnet', family = binomial(link='logit'),
  trControl = trainControl('cv', number=10, summaryFunction = mae_loss),
  tuneLength = 10,
  metric = "MAE"
)

# checking optimal tuning parameters and model coefficients with these parameters
model.net$bestTune # alpha = 0.1 lambda = 0.04462071
coef(model.net$finalModel, model.net$bestTune$lambda)


# fitting ridge regression with optimal lambda and alpha values and
# computing predicted values using validation dataset (as classess and probabilities of falling into classes)
set.seed(123)
modelbest.net <- glmnet(x_train, y_train, alpha = model.net$bestTune['alpha'],
                        lambda = model.net$bestTune['lambda'], family = binomial(link='logit'))

predictions.probs.net <- predict(modelbest.net, newx = x_validation, type = "response") %>% as.vector()

data.frame(
  RMSE = RMSE(predictions.probs.net, y_validation),
  Rsquare.r = R2(predictions.probs.net, y_validation))


# COMPARISON OF PERFORMANCE -------------------------------
# Table with comparison of RMSE and RSqared
perf.measurements <- rbind(MAE = c(round(MAE(predictions.probs.lasso, y_validation),5),
                                   round(MAE(predictions.probs.net, y_validation),5)),
                           R2 = c(round(R2(predictions.probs.lasso, y_validation),5),
                                  round(R2(predictions.probs.net, y_validation),5))
                           )
colnames(perf.measurements) <- c("Lasso regression", "Elastic nets")

as_tibble(perf.measurements, rownames='Metric')

write_csv(as_tibble(perf.measurements, rownames='Metric'), 'data/output/GML_performance.csv')

# Bootstrapping
set.seed(123)
random_starts <- sample(1:100000, 5000, replace = FALSE)
GLM_R2 <- c()
GLM_MAE <- c()

for(i in 1:length(random_starts)) {
  set.seed(random_starts[i])
  indices <- sample(1:nrow(x_validation), 300)
  x_validation_sample <- x_validation[indices,]
  y_validation_sample <- y_validation[indices]
  GLM_predictions <- predict(fit.lasso.min.lambda, newx = x_validation_sample, type = "response")
  GLM_MAE[i] <- MAE(GLM_predictions, y_validation_sample)
  GLM_R2[i] <- R2(GLM_predictions, y_validation_sample)
}

bootstrap_results <- tibble(GLM_R2, GLM_MAE)
write_csv(bootstrap_results, 'data/output/GLM_bootstrap_results.csv')



# Below code is just to recalculate the outputs of the logistic regression
#####################################################################

# Lasso double check ----
# Double check if lasso regression will work with only 1 column response variable corresponding to droupout percent

y_pred_check_function <- predict(fit.lasso.min.lambda, newx = x_train[1,], type = 'response')
y_pred_check_manual <- plogis(fit.lasso.min.lambda$a0+sum(x_train[1,] * fit.lasso.min.lambda$beta))
glue("Fitted value using GLM function is {round(y_pred_check_function,5)}:")
glue("Manual recalculation of the fitted value using coefficient of the model and logit function: {round(y_pred_check_manual,5)}")

# As glmnet takes families from stats package. I will manually recalculate log likelihood
# to check that log-loss is used for logistic regression with continuous response as it is used
# for classification proplems

# Checking log-likelihood of logistic regression for response with
simple_logistic <- glm(dropout_pct ~ ATTENDANCE_RATE, family = binomial(link = 'logit'), data = train)
simple_logistic_pred <- predict(simple_logistic, , newx = x_train, type = "response")
y_train_binary <- y_train
for(i in 1:length(y_train_binary)){
  if(y_train[i]<=0.5)
    y_train_binary[i] <- 0
  else
    y_train_binary[i] <- 1
}
logLik(simple_logistic)
sum((y_train_binary*log(simple_logistic_pred) + (1-y_train_binary)*log(1-simple_logistic_pred)))
# OK. log-loss is the log likelihood

#####################################################################
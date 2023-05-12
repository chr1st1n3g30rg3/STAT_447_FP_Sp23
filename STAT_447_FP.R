# STAT 447 Final Project Movie Data
# Christine George
# 5/7/2023

# Load packages
library(tidyverse)
library(ISLR2)
library(GGally)
library(dplyr)
library(leaps)
library(visreg)
library(bestglm)
library(glmnet)
library(ggplot2)
library(pls)
library(glmnet)
library(tree)
library(randomForest)
library(pROC)
library(mdsr)
library(caret)
library(rpart)
library(rpart.plot)

# Load movies dataset
Raw_Movies <- read.csv(file.choose())

# Clean data
missing_values <- is.na(Raw_Movies)
num_missing_values <- sum(missing_values)
# There are 2370 missing values
Clean_Movies <- na.omit(Raw_Movies)

# Split the data into training and test sets (70% and 30%, respectively)
set.seed(155) # set a seed for reproducibility
train_idx <- createDataPartition(Clean_Movies$gross, p = .7, list = FALSE)
train_data <- Clean_Movies[train_idx, ]
test_data <- Clean_Movies[-train_idx, ]
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# Fit the multiple linear regression model
model <- lm(gross ~ rating + genre + score + votes + budget + runtime,
            data = train_data)

# ASSUMPTIONS
# 1. Linearity
# 2. Independence: Observations are independent of each other
# 3. Normality: Random error follows a normal distribution
# 4. Equal-Variance: Random error has the same variance

# Check assumptions
# Linearity
plot(train_data$score, train_data$gross)
plot(train_data$votes, train_data$gross)
plot(train_data$budget, train_data$gross)
plot(train_data$runtime, train_data$gross)

# Normality
par(mfrow=c(2,2))
hist(model$residuals)
qqnorm(model$residuals)
qqline(model$residuals)
plot(model$fitted.values, model$residuals)

# Equal variance
# Create plot of residuals versus predicted values
plot(predict(model), resid(model),
     xlab = "Predicted values",
     ylab = "Residuals",
     main = "Residuals vs. Predicted values")
# Plot should have scattered data points and no distinct pattern, but the points
# on this residual plot are clustered together. Therefore, the assumption of 
# equal variance is not met.

# Regression formula
# gross = -6.766e+07 + 2.441e+04 * ratingG - 4.924e+06 * ratingNC-17
# + 2.549e+07 * ratingNot Rated + 2.641e+07 * ratingPG + 1.580e+07 * ratingPG-13
# - 2.836e+06 * ratingR + 3.104e+08 * ratingTV-MA + 2.412e+07 * ratingUnrated
# + 4.630e+06 * genreAdventure + 5.836e+07 * genreAnimation - 1.494e+07 * genreBiography
# + 1.126e+07 * genreComedy - 2.819e+06 * genreCrime - 1.168e+06 * genreDrama + 
# 4.680e+08 * genreFamily + 1.531e+07 * genreFantasy + 3.916e+07 * genreHorror
# - 4.210e+05 * genreMystery - 2.615e+07 * genreRomance - 3.531e+06 * genreSci-Fi
# + 3.595e+07 * genreThriller + 6.461e+06 * score + 3.592e+02 * votes
# + 2.547e+00 * budget - 1.820e+05 * runtime

# Convert genre and rating to factors
train_data$genre_factor <- factor(train_data$genre)
train_data$rating_factor <- factor(train_data$rating)

# Use the factors in the model
model <- lm(gross ~ budget + votes + rating_factor + score + runtime + genre_factor,
            data = train_data)
summary(model)

# Convert genre and rating in the test data to factors
test_data$genre_factor <- factor(test_data$genre)
test_data$rating_factor <- factor(test_data$rating)
test_data$genre_factor <- factor(test_data$genre, levels = levels(train_data$genre_factor))
test_data$rating_factor <- factor(test_data$rating, levels = levels(train_data$rating_factor))

# Generate predicted values
test_data$predicted_gross <- predict(model, newdata = test_data)

# Remove rows with missing values
test_data <- na.omit(test_data)

# Calculate mean squared error (MSE)
mse <- mean((test_data$gross - test_data$predicted_gross)^2)
mse

# Calculate R-squared
rsq <- cor(test_data$gross, test_data$predicted_gross)^2
rsq

# HYPOTHESES
# beta1 = regression coefficient for rating
# beta2 = regression coefficient for genre
# beta3 = regression coefficient for budget
# beta4 = regression coefficient for score
# beta5 = regression coefficient for votes
# beta6 = regression coefficient for runtime
# Null hypotheses: beta1 = 0, beta2 = 0, beta3 = 0, beta4 = 0, beta5 = 0, and
# beta6 = 0
# Alternative hypotheses: beta1 =/= 0, beta2 =/= 0, beta3 =/= 0, beta4 =/= 0,
# beta5 =/= 0, and beta6 =/= 0

# RESULTS
# REJECT null hypothesis for betas 1 through 5 and fail to reject the null for
# beta 6. Rating, genre, budget, score, and votes are helpful predictors of 
# revenue, but runtime is not.

# New model without runtime
new_model <- lm(gross ~ budget + votes + rating_factor + score + genre_factor,
                data = train_data)
summary(new_model)
coef(new_model)

# Generate predicted values
test_data$predicted_gross <- predict(new_model, newdata = test_data)

# Remove rows with missing values
test_data <- na.omit(test_data)

# Calculate mean squared error (MSE)
mse <- mean((test_data$gross - test_data$predicted_gross)^2)
mse

# Calculate R-squared
rsq <- cor(test_data$gross, test_data$predicted_gross)^2
rsq

# Create classification algorithm to classify movies as successful
# or unsuccessful, 1 means successful and 0 means unsuccessful
Clean_Movies$success <- ifelse(Clean_Movies$gross >= 3*Clean_Movies$budget,
                                  1, 0)

# Choose predictor variables
predictor_vars <- c("genre", "budget", "rating", "score")

# Prepare the data
Clean_Movies <- Clean_Movies[, c(predictor_vars, "success")]
Clean_Movies$genre <- as.factor(Clean_Movies$genre)
Clean_Movies$rating <- as.factor(Clean_Movies$rating)
Clean_Movies <- na.omit(Clean_Movies)

# Split the data
set.seed(333)
train_indices <- sample(nrow(Clean_Movies), nrow(Clean_Movies) * 0.7)
train_data <- Clean_Movies[train_indices, ]
test_data <- Clean_Movies[-train_indices, ]

# Convert genre and rating to factors
train_data$genre_factor <- factor(train_data$genre)
train_data$rating_factor <- factor(train_data$rating)

# Convert genre and rating in the test data to factors
test_data$genre_factor <- factor(test_data$genre)
test_data$rating_factor <- factor(test_data$rating)
test_data$genre_factor <- factor(test_data$genre, levels = levels(train_data$genre_factor))
test_data$rating_factor <- factor(test_data$rating, levels = levels(train_data$rating_factor))

# remove missing values from test_data
test_data <- na.omit(test_data)

# Train the model
model <- glm(success ~ genre_factor + rating_factor + budget + score,
             data = train_data, family = binomial)
summary(model)

# create predictions for test data
pred <- predict(model, newdata = test_data, type = "response")

# Convert predicted probabilities to binary classification (Yes/No)
pred_class <- ifelse(pred > 0.5, 1, 0)

# create confusion matrix
conf_mat <- table(test_data$success, pred_class)
conf_mat

# Calculate test error rate
test_error_rate <- sum(abs(pred_class - test_data$success)) / nrow(test_data)
test_error_rate

# Fit decision tree
tree.movies <- rpart(success ~ genre_factor + rating_factor + budget + score,
                     data = train_data, method = "class", cp = 0.01)
tree.pred <- predict(tree.movies, test_data,
                     type = "class")

# Make labels more intuitive
rpart.plot(tree.movies, type = 4, extra = 100)

# Create confusion matrix
tree_mat <- confusionMatrix(factor(tree.pred, levels = c("0", "1")),
                            factor(test_data$success, levels = c("0", "1")))
#             Reference
# Prediction    0    1
#          0 1055  381
#          1   77  117

# Calculate test error rate
tree_error <- (381+77)/1630
tree_error

# Define the cross-validation folds
folds <- createFolds(train_data$success, k = 10)

# Loop over the folds and compute the misclassification rate
misclass_rates <- numeric(length(folds))
for (i in seq_along(folds)) {
  # Extract the training and validation folds
  train_fold <- train_data[-folds[[i]], ]
  valid_fold <- train_data[folds[[i]], ]
  
  # Fit the tree model on the training fold
  tree.fold <- rpart(success ~ budget + genre_factor + rating_factor + score,
                     data = train_fold, method = "class")
  
  # Make predictions on the validation fold
  pred_fold <- predict(tree.fold, newdata = valid_fold, type = "class")
  
  # Compute the misclassification rate
  misclass_rates[i] <- mean(pred_fold != valid_fold$success)
}

# Compute the mean and standard deviation of the misclassification rates
mean(misclass_rates)
sd(misclass_rates)

# Random forest
set.seed(777)
rf.movies <- randomForest(success ~ budget + genre_factor + rating_factor
                          + score, data = train_data, ntree = 201,
                           mtry = sqrt(ncol(train_data) - 1))
print(rf.movies)
plot(rf.movies)

# Make predictions on test set
rf_pred <- predict(rf.movies, test_data, type = "class")
rfpred_class <- ifelse(rf_pred > 0.5, 1, 0)
rf_mat <- table(test_data$success, rfpred_class)
rf_mat

# Calculate test error rate
rf_error <- (297+169)/1630
rf_error

# Find the most important variable in the random forest
var_importance <- importance(rf.movies)
most_important_var <- rownames(var_importance)[which.max(var_importance)]
print(paste0("The most important variable is ", most_important_var))

# Calculate sensitivity and specificity for random forest
rf.sensitivity <- rf_mat[2, 2] / sum(rf_mat[2, ])
rf.specificity <- rf_mat[1, 1] / sum(rf_mat[1, ])
rf.sensitivity 
rf.specificity

# Create predictive model to predict IMDb ratings of movies
# Split the data into training and test sets (70% and 30%, respectively)
set.seed(235) # set a seed for reproducibility
train_idx <- createDataPartition(Clean_Movies$score, p = .7, list = FALSE)
train_data <- Clean_Movies[train_idx, ]
test_data <- Clean_Movies[-train_idx, ]
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# ASSUMPTIONS
# 1. Linearity
# 2. Independence: Observations are independent of each other
# 3. Normality: Random error follows a normal distribution
# 4. Equal-Variance: Random error has the same variance

# Check assumptions
# Linearity
plot(train_data$gross, train_data$score)
plot(train_data$votes, train_data$score)
plot(train_data$budget, train_data$score)
plot(train_data$runtime, train_data$score)

# Normality
par(mfrow=c(2,2))
hist(model2$residuals)
qqnorm(model2$residuals)
qqline(model2$residuals)
plot(model2$fitted.values, model2$residuals)

# Equal variance
plot(predict(model2), resid(model2),
     xlab = "Predicted values",
     ylab = "Residuals",
     main = "Residuals vs. Predicted values")

# Fit the multiple linear regression model
model2 <- lm(score ~ gross + rating + genre + votes + budget + runtime,
             data = train_data)
summary(model2)

# HYPOTHESES
# beta1 = regression coefficient for rating
# beta2 = regression coefficient for genre
# beta3 = regression coefficient for budget
# beta4 = regression coefficient for gross
# beta5 = regression coefficient for votes
# beta6 = regression coefficient for runtime
# Null hypotheses: beta1 = 0, beta2 = 0, beta3 = 0, beta4 = 0, beta5 = 0, and
# beta6 = 0
# Alternative hypotheses: beta1 =/= 0, beta2 =/= 0, beta3 =/= 0, beta4 =/= 0,
# beta5 =/= 0, and beta6 =/= 0

# RESULTS
# REJECT null hypothesis for betas 1 through 6. Rating, genre, budget, gross,
# votes, and runtime are all useful predictors of score.

test_data$predicted_score <- predict(model2, newdata = test_data)

# Calculate mean squared error (MSE)
mse <- mean((test_data$score - test_data$predicted_score)^2)
mse

# Calculate R-squared
rsq <- cor(test_data$score, test_data$predicted_score)^2
rsq

# Predict IMDb score for Everything Everywhere All at Once (2022)
EEAAO <- data.frame(budget = 14300000, genre = "Sci-Fi", runtime = 139,
                        rating = "R", votes = 440866, gross = 140200000)

# use the predict function to predict the score of the new movie
predicted_score <- predict(model2, EEAAO)
predicted_score







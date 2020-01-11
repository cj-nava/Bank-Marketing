library(tidyverse)
library(gmodels) #cross tables and logistic regression
library(caret)
library(ggplot2)
library(data.table)

# Read in data
df_raw <- read.csv("data/bank-additional-full.csv",sep = ";")

# Check for missing values
# Report the number of NA's in each column.
a = colnames(df_raw)
b = colSums(is.na(df_raw))  %>% as.data.table

missing_value_table = cbind(a, b)
colnames(missing_value_table) = c("Variables","Missing_values")
missing_value_table = missing_value_table  %>% 
  filter(Missing_values>0)  %>% 
  mutate("% of Total Values" = round(100 * (Missing_values / nrow(df_raw)), 1))  %>% 
  arrange(desc(Missing_values))

head(missing_value_table, 10)

# Report the number of "unknown"'s in each column.
a = colnames(df_raw)
b = colSums(df_raw=="unknown")  %>% as.data.table

unknown_value_table = cbind(a, b)
colnames(unknown_value_table) = c("Variables","Unknown_values")
unknown_value_table = unknown_value_table  %>% 
  filter(Unknown_values>0)  %>% 
  mutate("% of Total Values" = round(100 * (Unknown_values / nrow(df_raw)), 1))  %>% 
  arrange(desc(Unknown_values))

head(unknown_value_table, 10)

# The following EDA code based on and adapted from Kaggle user 
# Psqrt available from: https://www.kaggle.com/psqrtpsqrt/bank-marketing-eda-classification-pr-f-score

# Correlation Matrix of Continuous Variables
library(corrplot)
df_raw %>% 
  dplyr::select(emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) %>% 
  cor() %>% 
  corrplot(method = "number",
           type = "upper",
           tl.cex = 0.8,
           tl.srt = 45,
           tl.col = "black")

# Per the correlation matrix, there do not appear to be any variables of concern. We will keep all numeric variables.

# Age
df_raw %>% 
  ggplot() +
  aes(x = age) +
  geom_bar() +
  geom_vline(xintercept = c(30, 60), 
             col = "red",
             linetype = "dashed") +
  facet_grid(y ~ .,
             scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 100, 5))

# Cross Tables
CrossTable(df_raw$age, df_raw$y)          # Age
CrossTable(df_raw$job, df_raw$y)          # Job
CrossTable(df_raw$marital, df_raw$y)      # Marital Status    
CrossTable(df_raw$education, df_raw$y)    # Education
CrossTable(df_raw$default, df_raw$y)      # Default
CrossTable(df_raw$housing, df_raw$y)      # Housing
CrossTable(df_raw$loan, df_raw$y)         # Loan
CrossTable(df_raw$contact, df_raw$y)      # Contact
CrossTable(df_raw$month, df_raw$y)        # Month
CrossTable(df_raw$day_of_week, df_raw$y)  # Day of Week
CrossTable(df_raw$campaign, df_raw$y)     # Campaign
CrossTable(df_raw$pdays, df_raw$y)        # Pdays
CrossTable(df_raw$previous, df_raw$y)     # Previous
CrossTable(df_raw$poutcome, df_raw$y)     # Poutcome


# Split ages into three groups: "17-30", "30-60", ">60"
df_raw = df_raw %>% 
  mutate(age = if_else(age > 60, "high", if_else(age > 30, "mid", "low")))

# Remove observations with "unknown" values for job, marital, and education variables
df_raw = df_raw %>% 
  filter(job != "unknown") %>% 
  filter(marital != "unknown") %>% 
  filter(education != "unknown")

# Drop variables
df_raw = df_raw %>% 
  select(-c(duration, default, pdays, housing, loan))

# Set reference levels for factors
df_raw$contact <- factor(df_raw$contact, order = FALSE, levels =c('telephone', 'cellular'))
df_raw$education <- factor(df_raw$education, order = FALSE, levels =c('illiterate','basic.4y', 'basic.6y','basic.9y', 'high.school','professional.course','university.degree'))
df_raw$age <- factor(df_raw$age, order = FALSE, levels =c('low','mid', 'high'))
df_raw$job <- factor(df_raw$job, order = FALSE, levels =c('blue-collar', 'services','entrepreneur', 'housemaid', 'self-employed','technician', 'management','admin.','unemployed', 'retired','student'))
df_raw$marital <- factor(df_raw$marital, order = FALSE, levels =c('married', 'divorced', 'single'))
df_raw$month <- factor(df_raw$month, order = FALSE, levels =c('mar', 'apr','may', 'jun','jul', 'aug', 'sep','oct', 'nov','dec'))
df_raw$poutcome <- factor(df_raw$poutcome, order = FALSE, levels =c('nonexistent', 'failure','success'))
df_raw$y <- factor(df_raw$y, order = FALSE, levels = c('no', 'yes'))

# make copy of dataframe to avoid messing up original
df_bank.clean <- df_raw

# Split the data into training and test set
set.seed(123)
training.samples <- createDataPartition(df_bank.clean$y, p = 0.8, list = FALSE)
train.data  <- df_bank.clean[training.samples, ]
test.data <- df_bank.clean[-training.samples, ]

# deal with missing values: see missing data file in analysis folder for reference

bank <- map_df(bank, na_if, y = "unknown")
bank <- bank %>% drop_na(default, housing, loan, job, education, duration, marital) %>% filter(default == "no")
bank <- bank %>% dplyr::select(-c(default, duration, pdays))
bank <- bank %>% droplevels()
set.seed(193204)
#create training and test sets
idx <- createDataPartition(bank$y, p = 0.75, list = FALSE)
training.data <- bank[idx, ]
test.data <- bank[-idx, ]
balanced.training.data <- training.data %>% downsample(y)
balanced.training.data_X <- balanced.training.data %>% dplyr::select(-c(y))
test.data_X <- test.data %>% select(-c(y))

# Fit full model using all of the potential predictors
model_full <- glm(y ~ ., data = train.data, family = "binomial")
summary(model_full)

# Fit a model using the most relevant variables from the full model
model_final <- glm(y ~ contact + month + poutcome + emp.var.rate + cons.price.idx, 
                   family = binomial(link = 'logit'),
                   data = train.data)
summary(model_final)

# Coefficient confidence intervals
confint.default(model_final)

# odds ratios and 95% CI
exp(cbind(OR = coef(model_final), confint(model_final)))

library(caret)
predictions <- predict(model_final, newdata = test.data, type = "link")
predictions <- as.factor(map_chr(predictions, function(x){if(x >= 0.5){return("yes")} else{return("no")}}))

# confusion matrix
confusionMatrix(predictions, test.data$y, positive = "yes")

load("data/datasets.RData")

# Fit balanced model
model_balanced <- glm(y ~ contact + month + poutcome + emp.var.rate + cons.price.idx, 
                      family = binomial(link = 'logit'),
                      data = balanced.training.data)

# make predictions
predictions.balanced <- predict(model_balanced, newdata = test.data, type = "link")
predictions.balanced <- as.factor(map_chr(predictions.balanced, function(x){if(x >= 0.5){return("yes")} else{return("no")}}))

# confusion matrix
confusionMatrix(predictions.balanced, test.data$y, positive = "yes")

# Create prediction matrix
x = model.matrix(y~., balanced.training.data)[,-1]
y = balanced.training.data$y
xtest <- model.matrix(y~.,test.data)[,-1]
ytest <- test.data$y

# Fit ridge model
library(glmnet)
balanced.ridge.mod <- cv.glmnet(x, y, alpha = 0, type.measure = "class", nfolds = 5, family = "binomial")

# Lambda plot
plot(balanced.ridge.mod)
balanced.bestlambda <- balanced.ridge.mod$lambda.min
balanced.ridge.pred <- as.factor(predict(balanced.ridge.mod, s = balanced.bestlambda, newx = xtest, type = "class"))

# confusion matrix
confusionMatrix(data = balanced.ridge.pred, reference = ytest, positive = "yes")

library(MASS)
# Fit LDA Model
lda.model.balanced <- lda(y ~ contact + month + poutcome + emp.var.rate + cons.price.idx,
                          data = balanced.training.data)

# Make predictions
predmodel.train.lda = predict(lda.model.balanced, newdata = test.data, type = "class")

# Confusion matrix
confusionMatrix(data = predmodel.train.lda$class, reference = test.data$y, positive = "yes")

library(randomForest)
library(naniar)
load("data/datasets.RData")

library(FactoMineR)
features <- bank %>% select(-c(y))
model <- randomForest(y ~ ., data = bank, mtry  = 1, importance = TRUE)
model2 <- randomForest(y ~ ., mtry = 1, data = bank, importance = TRUE)


source("../src/featureImportance/MDA.R")
source("../src/featureImportance/MDI.R")
plotImpMDI(model)

source("../src/multicollinearity/conditionNumber.R")
paste("Condition Number:",conditionNumber(features))

Low condition number suggests that we do not have to worry about multicollinearity.

plotImpMDA(model2)

library(caret)
library(doMC)
registerDoMC(cores = 4)
set.seed(825)
oob = trainControl(method = "oob")
rf_grid =  expand.grid(mtry = 1:18, splitrule = "gini", min.node.size = 500)
balanced_rf_model = train(y ~ ., data = balanced.training.data,
                          method = "ranger",
                          trControl = oob,
                          verbose = FALSE,
                          tuneGrid = rf_grid,
                          metric = "Accuracy")
plot(balanced_rf_model)

new_balanced_rf_model <- randomForest(x = balanced.training.data_X, y = balanced.training.data$y, mtry = 15, nodesize = 25)

balanced.pred <- predict(new_balanced_rf_model, newdata = test.data)
confusionMatrix(data = balanced.pred, reference = test.data$y, positive = "yes")
new_balanced_rf_model.roc <- roc(test.data$y, as.ordered(balanced.pred))
ggroc(new_balanced_rf_model.roc) + labs(title = "Random Forest ROC (Balanced)",caption = paste("AUC:",round(new_balanced_rf_model.roc$auc, 3))) 

logistic_scatter <- function(data, response, ..., smoothing = 0.7, jitter=FALSE){
  predictors <- map(enexprs(...), function(x){return(expr(`$`(!!data, !!(x))))})
  response <- expr(`$`(!!enexpr(data), !!enexpr(response)))
  graph_list <- map(predictors, loess_plot, response = response, span = smoothing, jitter=jitter)
  names(graph_list) <- as.character(eval(enexprs(...)))
}
loess_plot <- function(predictor, response, span, jitter){
  response <- as.numeric(eval(response)) - 1
  loess.mod <- loess(response ~ eval(predictor), span = span)
  loess.pred <- predict(loess.mod)
  prob <- pmax(pmin(loess.pred, 0.9999), 0.0001)
  logit.loess.pred <- logit(prob)
  if(jitter == FALSE){
    return(ggplot(NULL,aes(x = eval(predictor), y = logit.loess.pred)) + geom_point() +
             labs(y = "Estimated logit(p)", x = predictor))
  }
  else{
    return(ggplot(NULL,aes(x = predictor, y = logit.loess.pred)) + geom_point(position = "jitter") +
             labs(y = "Estimated logit(p)", x = predictor))
  }
}
logit <- function(p){
  log(p/(1-p))
}

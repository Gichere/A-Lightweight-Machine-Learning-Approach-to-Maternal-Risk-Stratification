
# Load libraries ----------------------------------------------------------------------------------------------------------------

pacman::p_load(tidyverse, caret, pROC, rpart, rpart.plot, ggplot2, tidyverse, readr)

# Get UCL data ------------------------------------------------------------------------------------------------------------------

# Define URL and temporary paths

zip_url <- "https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip"

temp_zip <- tempfile(fileext = ".zip")

temp_dir <- tempdir()

# Download and unzip

download.file(zip_url, temp_zip)

unzip(temp_zip, exdir = temp_dir)

# List files to find the CSV

list.files(temp_dir)

# Load the dataset

csv_file <- file.path(temp_dir, "Maternal Health Risk Data Set.csv")

data <- read_csv(csv_file)

# Inspect data ------------------------------------------------------------------------------------------------------------------

str(data)

summary(data)

# Recode RiskLevel to binary for simplicity (low vs. medium/high) ---------------------------------------------------------------

data$RiskLevelBinary <- ifelse(data$RiskLevel == "low risk", 0, 1) # 0 = low, 1 = medium/high

# Check for missing values ------------------------------------------------------------------------------------------------------

colSums(is.na(data)) # No missing values in this dataset

# Normalize continuous features -------------------------------------------------------------------------------------------------

preProcess <- preProcess(data[, c("Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate")], method = c("center", "scale"))

data_normalized <- predict(preProcess, data)

# Split into training and test sets (80/20) -------------------------------------------------------------------------------------

set.seed(123)

trainIndex <- createDataPartition(data_normalized$RiskLevelBinary, p = 0.8, list = FALSE)

train_data <- data_normalized[trainIndex, ]

test_data <- data_normalized[-trainIndex, ]

# Train logistic regression -----------------------------------------------------------------------------------------------------

log_model <- glm(RiskLevelBinary ~ Age + SystolicBP + DiastolicBP + BS + BodyTemp + HeartRate, data = train_data, family = binomial)

# Predict on test set

log_pred_prob <- predict(log_model, test_data, type = "response")

log_pred <- ifelse(log_pred_prob > 0.5, 1, 0)

# Train decision tree -----------------------------------------------------------------------------------------------------------

tree_model <- rpart(RiskLevelBinary ~ Age + SystolicBP + DiastolicBP + BS + BodyTemp + HeartRate, data = train_data, method = "class", 
                    
                    control = rpart.control(maxdepth = 5))

# Predict on test set

tree_pred <- predict(tree_model, test_data, type = "class")

# Evaluation --------------------------------------------------------------------------------------------------------------------

# Logistic Regression 

log_confusion <- confusionMatrix(factor(log_pred), factor(test_data$RiskLevelBinary), positive = "1")

log_roc <- roc(test_data$RiskLevelBinary, log_pred_prob)

# Decision Tree Evaluation

tree_confusion <- confusionMatrix(factor(tree_pred), factor(test_data$RiskLevelBinary), positive = "1")

tree_roc <- roc(test_data$RiskLevelBinary, as.numeric(tree_pred))

# Print results
print("Logistic Regression Performance:")

print(log_confusion)

print(paste("AUC:", auc(log_roc)))

print("Decision Tree Performance:")

print(tree_confusion)

print(paste("AUC:", auc(tree_roc)))

# Plot ROC curves for both models -----------------------------------------------------------------------------------------------

roc_data <- data.frame(Model = c(rep("Logistic Regression", length(log_roc$sensitivities)), 
                                 
                                 rep("Decision Tree", length(tree_roc$sensitivities))),
                       
                       Sensitivity = c(log_roc$sensitivities, tree_roc$sensitivities), 
                       
                       Specificity = c(1 - log_roc$specificities, 1 - tree_roc$specificities))

ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Model)) +
  
  geom_line(size = 1) +
  
  geom_abline(linetype = "dashed", color = "gray") +
  
  labs(title = "ROC Curves for Maternal Risk Classifiers",   x = "1 - Specificity", y = "Sensitivity") +
  
  theme_minimal() +
  
  scale_color_manual(values = c("Logistic Regression" = "blue", "Decision Tree" = "red"))

ggsave("roc_curve.png", width = 6, height = 4)

# Extract coefficients from logistic regression ---------------------------------------------------------------------------------

coef_data <- data.frame(Feature = names(coef(log_model))[-1], Coefficient = coef(log_model)[-1])

ggplot(coef_data, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  
  geom_bar(stat = "identity", fill = "steelblue") +
  
  coord_flip() +
  
  labs(title = "Feature Importance in Logistic Regression", x = "Feature", y = "Coefficient") +
  
  theme_minimal()

ggsave("logistic_feature_importance.png", width = 6, height = 4)

# Plot decision tree ------------------------------------------------------------------------------------------------------------

png("decision_tree.png", width = 6, height = 4, units = "in", res = 300)

rpart.plot(tree_model, main = "Decision Tree for Maternal Risk Stratification", box.palette = "Blues",  shadow.col = "gray")

dev.off()

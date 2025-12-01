
# ---------------------------------------------------------
# DSCI 724 Project – Insurance Dataset R Script
# Student: Teresia Wainaina
# ---------------------------------------------------------
# This script tests the insurance dataset against the
# DSCI 724 project rubric requirements.
# ---------------------------------------------------------

# 1. Clear workspace
rm(list = ls())

# 2. Setting the working directory
setwd("C:/Users/Teresia/Documents")  # Change if needed
getwd()

# Creating an "outputs" folder if it doesn’t exist
if (!dir.exists("outputs")) dir.create("outputs")

# 3. Installing and loading required libraries
packages <- c("tidyverse", "psych", "corrplot", "caret", "pROC")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# 4. Loading dataset
insurance <- read_csv("insurance[1].csv")  # Make sure the file exists

# 5. Basic info
print("Dataset loaded successfully")
print(paste("Rows:", nrow(insurance)))
print(paste("Columns:", ncol(insurance)))

head(insurance)
str(insurance)
summary(insurance)

# 6. Cleaning missing data
insurance <- insurance %>% drop_na()
print("Missing values removed (if any).")

# 7. Identifying variable types
numeric_vars <- insurance %>% select(where(is.numeric))
categorical_vars <- insurance %>% select(where(~is.character(.) | is.factor(.)))

print("Numeric variables:")
print(colnames(numeric_vars))
print("Categorical variables:")
print(colnames(categorical_vars))

# 8. Checking dataset size
if (nrow(insurance) < 100) {
  print("Dataset has fewer than 100 records.")
} else {
  print("Dataset meets 100+ record requirement.")
}

# ---------------------------------------------------------
# 9. Descriptive Statistics
# ---------------------------------------------------------
if (ncol(numeric_vars) > 0) {
  print("Descriptive Statistics (numeric variables):")
  describe(numeric_vars)
  
  means <- sapply(numeric_vars, mean, na.rm = TRUE)
  sds <- sapply(numeric_vars, sd, na.rm = TRUE)
  print("Variable Means:")
  print(means)
  print("Variable Standard Deviations:")
  print(sds)
} else {
  print("No numeric variables to describe.")
}

# ---------------------------------------------------------
# 10. Boxplots for numeric variables (Display + Save)
# ---------------------------------------------------------
if (ncol(numeric_vars) > 0) {
  boxplot_plot <- numeric_vars %>%
    pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
    ggplot(aes(x = Variable, y = Value, fill = Variable)) +
    geom_boxplot() +
    theme_minimal() +
    ggtitle("Boxplots of Numeric Variables") +
    theme(legend.position = "none")
  
  # Displaying on console
  print(boxplot_plot)
  
  # Saving as PNG
  png(filename = "outputs/boxplots_numeric.png", width = 900, height = 600)
  print(boxplot_plot)
  dev.off()
  print("Boxplot image saved: outputs/boxplots_numeric.png")
} else {
  print("No numeric variables to plot.")
}

# Loop over numeric variables and create a separate boxplot for each
for (var in colnames(numeric_vars)) {
  p <- ggplot(insurance, aes_string(y = var)) +
    geom_boxplot(fill = "turquoise") +
    theme_minimal() +
    ggtitle(paste("Boxplot of", var))
  
  # Display in console
  print(p)
  
  # Save each boxplot
  png(filename = paste0("outputs/boxplot_", var, ".png"), width = 600, height = 400)
  print(p)
  dev.off()
}

####Faceted boxplot (all in one figure but separate panels)

numeric_vars_long <- numeric_vars %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

ggplot(numeric_vars_long, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot() +
  theme_minimal() +
  facet_wrap(~Variable, scales = "free") +
  ggtitle("Boxplots of Numeric Variables (Separated Panels)") +
  theme(legend.position = "none")


# ---------------------------------------------------------
# 11. Correlation Matrix and Collinearity (Display + Save)
# ---------------------------------------------------------
if (ncol(numeric_vars) > 1) {
  cor_matrix <- cor(numeric_vars)
  print("Pearson Correlation Matrix:")
  print(round(cor_matrix, 2))
  
  # Showing in console
  corrplot(cor_matrix, method = "number", type = "upper")
  
  # Saving graphic version
  png(filename = "outputs/correlation_matrix.png", width = 800, height = 800)
  corrplot(cor_matrix, method = "number", type = "upper")
  dev.off()
  print("Correlation matrix saved: outputs/correlation_matrix.png")
  
  # Checking collinearity
  high_corr <- which(abs(cor_matrix) > 0.9 & abs(cor_matrix) < 1, arr.ind = TRUE)
  high_corr <- high_corr[high_corr[,1] < high_corr[,2], ]  # remove duplicate pairs
  
  if (nrow(high_corr) == 0) {
    print("No collinearity issues detected (no correlations > 0.9)")
  } else {
    print("High correlation detected between:")
    print(high_corr)
  }
} else {
  print("Not enough numeric columns for correlation or collinearity check.")
}

# ---------------------------------------------------------
# 11.5 Train/Test Split (80/20)
# ---------------------------------------------------------

set.seed(123)  # Ensures reproducibility

train_index <- createDataPartition(insurance$charges, p = 0.8, list = FALSE)

train_data <- insurance[train_index, ]
test_data  <- insurance[-train_index, ]

print(paste("Training rows:", nrow(train_data)))
print(paste("Testing rows:", nrow(test_data)))




# 12. Fit linear regression model with interactions
# Response: charges
# Predictors: age, sex, bmi, children, smoker, region
# Interaction terms: age:bmi, smoker:bmi
model <- lm(charges ~ age + sex + bmi + children + smoker + region + age:bmi + smoker:bmi,
            data=train_data)
summary(model)

# 13. Evaluate model: R², Adj R², RMSE, MAE, AIC, BIC
preds <- predict(model, newdata = test_data)
resid <- test_data$charges - preds

# R² manually computed on test set
SSE <- sum(resid^2)
SST <- sum((test_data$charges - mean(test_data$charges))^2)
R2 <- 1 - SSE/SST

# Adjusted R² for test set (optional)
Adj_R2 <- 1 - (1 - R2) * ((nrow(test_data) - 1) / (nrow(test_data) - length(model$coefficients)))

# 13. Evaluate model on TEST SET
preds <- predict(model, newdata = test_data)
resid <- test_data$charges - preds

SSE <- sum(resid^2)
SST <- sum((test_data$charges - mean(test_data$charges))^2)

R2 <- 1 - SSE/SST
Adj_R2 <- 1 - (1 - R2) * ((nrow(test_data) - 1) / (nrow(test_data) - length(model$coefficients)))

RMSE <- sqrt(mean(resid^2))
MAE <- mean(abs(resid))
AIC_val <- AIC(model)  # From training model
BIC_val <- BIC(model)

metrics <- data.frame(
  Metric = c("R2 (Test)", "Adj_R2 (Test)", "RMSE (Test)", "MAE (Test)", "AIC (Train)", "BIC (Train)"),
  Value  = c(R2, Adj_R2, RMSE, MAE, AIC_val, BIC_val)
)

write_csv(metrics, "outputs/evaluation_metrics2.csv")
print(metrics)
  

# 14. Diagnostic plots
png("outputs/residuals_vs_fitted.png", width=800, height=600)
plot(model, which=1)
dev.off()

png("outputs/normal_qq.png", width=800, height=600)
plot(model, which=2)
dev.off()

png("outputs/scale_location.png", width=800, height=600)
plot(model, which=3)
dev.off()

png("outputs/residuals_vs_leverage.png", width=800, height=600)
plot(model, which=5)
dev.off()

# 15. VIF Analysis
# Full model with interactions
# Install car package if not already installed
#install.packages("car")
# Load the package
library(car)
# vif_full <- vif(model)
vif_full <- vif(model, type = "predictor")
print(vif_full)
#write_csv(data.frame(Variable=names(vif_full), VIF=vif_full), "outputs/vif_full_model.csv")

# Separate model without interactions (required by rubric)
vif_model <- lm(charges ~ age + sex + bmi + children + smoker + region, data=insurance)
vif_no_inter <- car::vif(vif_model)
# write_csv(data.frame(Variable=names(vif_no_inter), VIF=vif_no_inter), "outputs/vif_no_interactions.csv")
print("VIF values (no interactions):")
print(vif_no_inter)

# 16. Individual predictor plots vs response
# Numeric predictors
for(var in c("age","bmi","children")){
  p <- ggplot(test_data, aes_string(x=var, y="charges")) +
    geom_point(alpha=0.6) +
    geom_smooth(method="lm", col="blue") +
    theme_minimal() +
    ggtitle(paste("Charges vs", var))
  print(p)
  ggsave(paste0("outputs/charges_vs_", var, ".png"), plot=p, width=8, height=6)
}

# Categorical predictors
for(var in c("sex","smoker","region")){
  p <- ggplot(test_data, aes_string(x=var, y="charges")) +
    geom_boxplot(fill="lightblue") +
    theme_minimal() +
    ggtitle(paste("Charges by", var))
  print(p)
  ggsave(paste0("outputs/charges_by_", var, ".png"), plot=p, width=8, height=6)
}

# 17. Save final model and residuals (optional)
saveRDS(model, "outputs/final_model.rds")
residuals_df <- data.frame(Observed=test_data$charges, Predicted=preds, Residuals=resid)
write_csv(residuals_df, "outputs/residuals.csv")

cat("All outputs saved in 'outputs/' folder. Script complete.\n")


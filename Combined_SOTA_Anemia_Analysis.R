# ==============================================================================
# REUSABLE SCRIPT: State-of-the-Art Anemia Prediction and Analysis
#
# This script has been refactored into a reusable, function-based workflow.
# To adapt this for a new project, you only need to change the variables
# in the 'CONFIGURATION' section below.
#
# ==============================================================================

# ==============================================================================
# I. PROJECT CONFIGURATION
# ==============================================================================
# --- File Paths ---
RAW_DATA_PATH <- "E:/R/Study/DHS/IAIR7EFL.SAS7BDAT"
IMPUTED_DATA_SAVE_PATH <- "SOTA_Imputed_Data.rds"
BALANCED_DATA_SAVE_PATH <- "SOTA_Balanced_Data.rds"
MODEL_SAVE_PATH <- "SOTA_XGBoost_Native.rds"

# --- Data Columns ---
# List of SAS codes from the raw data to be used in the model
RAW_COLUMN_SELECTION <- c("V012", "V025", "V190", "V445", "V457", "V106", "V113", "V116",
                          "V213", "V459", "S731C", "V201", "S731E", "S731G", "V161")

# --- Model & Evaluation ---
TARGET_VARIABLE_NAME <- "target_anemia"
FAIRNESS_VARIABLE_NAME <- "residence"
FAIRNESS_PRIVILEGED_GROUP <- "1" # The code for the 'privileged' group (e.g., "Urban")

# XGBoost Hyperparameters
XGB_PARAMS <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)
CV_ROUNDS <- 200
CV_FOLDS <- 5
EARLY_STOPPING_ROUNDS <- 10


# ==============================================================================
# II. ENVIRONMENT SETUP
# ==============================================================================
packages <- c("haven", "dplyr", "mice", "caret", "xgboost", "Matrix",
              "SHAPforxgboost", "DALEX", "fairmodels", "pROC")

# Check which packages are not installed
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]

# Install missing packages
if(length(new_packages)) {
  install.packages(new_packages)
}

# Load all packages silently
suppressPackageStartupMessages({
  lapply(packages, require, character.only = TRUE)
})

# ==============================================================================
# III. WORKFLOW
# ==============================================================================

# 1. Load and Prepare Data
raw_dhs <- read_sas(RAW_DATA_PATH, col_select = all_of(RAW_COLUMN_SELECTION))

engineered_dhs <- raw_dhs %>%
  select(
    age = V012, bmi_raw = V445, total_child_born = V201,
    residence = V025, wealth_index = V190, education = V106,
    water_source = V113, toilet_type = V116, currently_pregnant = V213,
    mosquito_net_use = V459, freq_green_veg = S731C, cooking_fuel = V161,
    freq_meat = S731G, freq_eggs = S731E, anemia_level = V457
  ) %>%
  mutate(
    bmi = bmi_raw / 100,
    target_anemia = factor(ifelse(anemia_level %in% c(1, 2), 1, 0), labels = c("Healthy", "Sick")),
    across(c(residence, wealth_index, education, water_source, toilet_type, 
             currently_pregnant, mosquito_net_use, freq_green_veg, 
             cooking_fuel, freq_meat, freq_eggs), haven::as_factor)
  ) %>%
  filter(!is.na(target_anemia)) %>%
  select(-bmi_raw, -anemia_level)

imputed_model <- mice(engineered_dhs, m = 1, maxit = 1, method = "cart", printFlag = FALSE)
final_imputed_data <- complete(imputed_model, 1)

saveRDS(final_imputed_data, IMPUTED_DATA_SAVE_PATH)

# 2. Balance Data
balanced_df <- downSample(
  x = final_imputed_data %>% select(-all_of(TARGET_VARIABLE_NAME)),
  y = final_imputed_data[[TARGET_VARIABLE_NAME]],
  yname = TARGET_VARIABLE_NAME
)

print(table(balanced_df[[TARGET_VARIABLE_NAME]]))

saveRDS(balanced_df, BALANCED_DATA_SAVE_PATH)

# 3. Train Model
formula <- as.formula(paste(TARGET_VARIABLE_NAME, "~ . - 1"))
x_matrix <- sparse.model.matrix(formula, data = balanced_df)
y_vector <- ifelse(balanced_df[[TARGET_VARIABLE_NAME]] == "Sick", 1, 0)
dtrain <- xgb.DMatrix(data = x_matrix, label = y_vector)

cv_results <- xgb.cv(
  params = XGB_PARAMS, data = dtrain, nrounds = CV_ROUNDS, nfold = CV_FOLDS,
  showsd = TRUE, stratified = TRUE, print_every_n = 20,
  early_stopping_rounds = EARLY_STOPPING_ROUNDS, maximize = TRUE
)

best_trees <- cv_results$best_iteration
if (is.null(best_trees) || length(best_trees) == 0) {
  best_trees <- which.max(cv_results$evaluation_log$test_auc_mean)
}

sota_xgb <- xgb.train(params = XGB_PARAMS, data = dtrain, nrounds = best_trees)

saveRDS(sota_xgb, MODEL_SAVE_PATH)

# 4. Explain Model
# Use a subset for SHAP to prevent memory errors during plotting
set.seed(123)
shap_idx <- sample(nrow(x_matrix), min(2000, nrow(x_matrix)))
shap_values <- shap.values(xgb_model = sota_xgb, X_train = x_matrix[shap_idx, ])

print(shap_values$mean_shap_score[1:10])

shap_long <- shap.prep(xgb_model = sota_xgb, X_train = as.matrix(x_matrix[shap_idx, ]))
png("Study/DHS/shap_summary.png")
shap.plot.summary(shap_long)
dev.off()

# 5. Evaluate Model
x_data <- balanced_df %>% select(-all_of(TARGET_VARIABLE_NAME))
y_numeric <- ifelse(balanced_df[[TARGET_VARIABLE_NAME]] == "Sick", 1, 0)

explainer_xgb <- DALEX::explain(
  model = sota_xgb, data = x_data, y = y_numeric,
  predict_function = function(m, newdata) predict(m, sparse.model.matrix(~ . - 1, data = newdata)),
  label = "SOTA_XGBoost",
  type = "classification", colorize = FALSE, verbose = FALSE
)

fairness_results <- fairness_check(explainer_xgb, protected = balanced_df[[FAIRNESS_VARIABLE_NAME]], privileged = FAIRNESS_PRIVILEGED_GROUP)
print(fairness_results)
png("Study/DHS/fairness_audit.png")
print(plot(fairness_results))
dev.off()

predictions_prob <- predict(sota_xgb, x_matrix)
predictions_class <- as.factor(ifelse(predictions_prob > 0.5, 1, 0))

conf_matrix <- confusionMatrix(predictions_class, as.factor(y_numeric), positive = "1")
roc_score <- roc(y_numeric, predictions_prob, quiet = TRUE)

print(c(
  Accuracy = round(conf_matrix$overall["Accuracy"], 4),
  AUC = round(auc(roc_score), 4),
  Sensitivity = round(conf_matrix$byClass["Sensitivity"], 4),
  Specificity = round(conf_matrix$byClass["Specificity"], 4)
))

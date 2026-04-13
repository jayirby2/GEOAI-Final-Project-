## Analysis ##

## Performance metrics matrix ## 

## Resample (CV) results ##
model_rr <- list(
  "XGBoost" = rr_xgb,
  "Random Forest" = rr_rf,
  "GAM" = rr_gam,
  "Logistic" = rr_glm_spatial,
  "SVM" = rr_svm
)

## Performance Matrix ##
results_df <- do.call(rbind, lapply(names(model_rr), function(name) {
  rr <- model_rr[[name]]
  
  data.frame(
    Model     = name,
    AUC       = rr$aggregate(msr("classif.auc")),
    LogLoss   = rr$aggregate(msr("classif.logloss")),
    Brier     = rr$aggregate(msr("classif.bbrier")),
    Precision = rr$aggregate(msr("classif.precision")),
    Recall    = rr$aggregate(msr("classif.recall"))
  )
})) %>% 
  arrange(desc(AUC)) %>%
            as.data.frame()

## Calibration analysis - do my predicted probabilities actually match reality ##

## Calibration function ##
# Assesses calibration for a given model's resampling result by comparing 
# predicted probability bins to actual frequency
assess_calibration <- function(resampling_result) {
  # Out of sample predictions
  pred <- resampling_result$prediction()
  calibration_df <- data.frame(
    prob = pred$prob[, "1"],
    truth = as.numeric(pred$truth == "1")
  ) %>%
    mutate(bin = cut(prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(prob),
      mean_obs  = mean(truth),
      n = n()
    )
  
  return(calibration_df)
}

xgb_calib <- assess_calibration(rr_xgb)
rf_calib <- assess_calibration(rr_rf)
gam_calib <- assess_calibration(rr_gam)
glm_calib <- assess_calibration(rr_glm_spatial)
svm_calib <- assess_calibration(rr_svm)

## Feature Analysis ##
## Importance evaluated with 'permutation' across all CV folds ##
imp_list <- lapply(rr_rf$learners, function(l) l$model$variable.importance)
imp_avg <- Reduce("+", imp_list) / length(imp_list)



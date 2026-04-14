## Assign a spatial task ##
task <- TaskClassifST$new(
  id = "bip",
  backend = data,
  target = "outcome",
  positive = "1",
  coordinate_names = c("hc_x", "hc_y")
)
task$set_col_roles(c("hc_x", "hc_y"), add_to = "feature")

## Spatial CV -- Need nested for models with hyperparameters## 
resampling_outer <- rsmp("spcv_block", folds = 5, rows = 5, cols = 5)
resampling_inner <- rsmp("spcv_block", folds = 5, rows = 5, cols = 5)


#############################################
## Logistic Regression learner ##
learner_glm <- lrn("classif.log_reg", predict_type = "prob")

# Resample Result object (runs CV)
rr_glm_spatial <- resample(task, learner_glm, resampling_outer)
(rr_glm_spatial_auc <- rr_glm_spatial$aggregate(msr("classif.auc")))
(rr_glm_spatial_logloss <- rr_glm_spatial$aggregate(msr("classif.logloss")))

## Compare to random CV ##
resampling_rand <- rsmp("cv", folds = 5)
resampling_rand$instantiate(task)
# Resample Result object (runs CV)
rr_glm_rand <- resample(task, learner_glm, resampling_rand)
(rr_glm_rand_auc <- rr_glm_rand$aggregate(msr("classif.auc")))
(rr_glm_rand_logloss <- rr_glm_rand$aggregate(msr("classif.logloss")))

## Comparing spatial and random CV ##
(comparing_CV_GLM <- data.frame(
  Method = c("Spatial CV", "Random CV"),
  AUC = c(rr_glm_spatial_auc, rr_glm_rand_auc),
  LogLoss = c(rr_glm_spatial_logloss, rr_glm_rand_logloss)
))

## Given the spatial coordinates and how close they are (field isn't that "big")
## I will be using spatial CV ##

############################################################

## Generalized Additive Model (GAM) using a learner ##
learner_gam <- lrn("classif.gam", predict_type = "prob")

# Fitting basis functions (will mess around with this)
learner_gam$param_set$values$formula <- outcome ~ s(hc_x, hc_y, bs = "tp") +
  s(launch_speed) +
  s(launch_angle) +
  if_fielding_alignment +
  of_fielding_alignment

rr_gam <- resample(task, learner_gam, resampling_outer)

###############################################

## Random Forest ##
learner_rf <- lrn("classif.ranger", predict_type = "prob", 
                  importance = "permutation")

search_space_rf <- ps(
  mtry = p_int(1, length(task$feature_names)),
  min.node.size = p_int(1, 10)
)
at_rf <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_rf,
  resampling = resampling_inner,
  measure = msr("classif.auc"),
  search_space = search_space_rf,
  terminator = trm("evals", n_evals = 10),
  store_models = FALSE
)

rr_rf <- resample(task, at_rf, resampling_outer, store_models = TRUE)

# Metrics
rr_rf$aggregate(msr("classif.auc"))
rr_rf$aggregate(msr("classif.logloss"))

###########################################################

## Boosted Model (XGBoost) ##
## One-hot encode manually to avoid errors
X <- model.matrix(outcome ~ . - 1, data = data)

data_encoded <- as.data.frame(X)
data_encoded$outcome <- data$outcome
names(data_encoded) <- make.names(names(data_encoded))
task <- TaskClassifST$new(
  id = "bip",
  backend = data_encoded,
  target = "outcome",
  positive = "1",
  coordinate_names = c("hc_x", "hc_y")
)

task$set_col_roles(c("hc_x", "hc_y"), add_to = "feature")

learner_xgb <- lrn("classif.xgboost", predict_type = "prob")

## Tuning Hyperparameters ##
search_space_xgb <- ps(
  eta = p_dbl(0.05, 0.15),
  max_depth = p_int(4, 6),
  subsample = p_dbl(0.7, 0.9),
  colsample_bytree = p_dbl(0.7, 0.9),
  nrounds = p_int(80, 150)
)

at_xgb <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_xgb,
  resampling = resampling_inner,
  measure = msr("classif.auc"),
  search_space = search_space_xgb,
  terminator = trm("evals", n_evals = 10),
  store_models = FALSE
)

rr_xgb <- resample(task, at_xgb, resampling_outer)

rr_xgb$aggregate(msr("classif.auc"))
rr_xgb$aggregate(msr("classif.logloss"))


###########################################################

## SVM ##
learner_svm <- po("scale") %>>% ## Making sure to scale variables here ## 
  lrn(
    "classif.svm",
    predict_type = "prob",
    kernel = "radial",
    type = "C-classification"
  )

## Search space ##
search_space_svm <- ps(
  classif.svm.cost = p_dbl(0.1, 10, logscale = TRUE),
  classif.svm.gamma = p_dbl(1e-4, 1, logscale = TRUE)
)

## Auto tuner ##
at_svm <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_svm,
  resampling = resampling_inner,
  measure = msr("classif.auc"),
  search_space = search_space_svm,
  terminator = trm("evals", n_evals = 10),
  store_models = FALSE
)
rr_svm <- resample(task, at_svm, resampling_outer)
rr_svm$aggregate(msr("classif.auc"))
rr_svm$aggregate(msr("classif.logloss"))



## Main Pipeline ##

############################################

## Reproducibility ##
set.seed(123)

## Packages ##
library(tidyverse)
library(mlr3)
library(mlr3learners)
library(mlr3spatiotempcv)
library(mlr3pipelines)
library(mlr3tuning)
library(paradox)
library(gt)
library(ggplot2)
library(shapviz)
library(data.table)

# =========================
# STEP 1: DATA
# =========================
source("DataProcessing.R")

## creates: data ##

## STEP 2: MODELING ##
source("Modeling.R")

# creates:
# rr_xgb, rr_rf, rr_gam, rr_glm_spatial, rr_svm

## STEP 3: ANALYSIS ##
source("Analysis.R")

# creates:
# results_df, calibration tables, imp_avg

## STEP 4: VISUALIZATION ##
source("Visualizations.R")

# creates:
# pt (gt table), p_hit_hm, shap, etc.

## STEP 5: SAVE OUTPUTS ##

dir.create("output", showWarnings = FALSE)

gtsave(pt, "output/model_performance.png")
gtsave(pt_imp, "output/rf_importance.png")

ggsave("output/heatmap.png", plot = p_hit_hm, width = 8, height = 8)

# Optional calibration saves (can save any of the model's calibration report)
# gtsave(xgb_calib_viz, "output/xgb_calib.png")
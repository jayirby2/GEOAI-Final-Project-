library(gt)
library(ggplot2)
library(shapviz)

## Visualizations ##

## Evalutation Metrics ## 

pt <- results_df %>%
  gt() %>%
  tab_header(
    title = md("**Model Performance Summary**"),
    subtitle = "Out-of-sample evaluation metrics"
  ) %>%
  fmt_number(
    columns = where(is.numeric),
    decimals = 3
  ) %>%
  data_color(
    columns = AUC,
    colors = scales::col_numeric(
      palette = c("#d73027", "#fee08b", "#1a9850"),
      domain = NULL
    )
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      columns = AUC,
      rows = AUC == max(AUC, na.rm = TRUE)
    )
  ) %>%
  cols_align(
    align = "center",
    everything()
  ) %>%
  opt_row_striping() %>%
  tab_options(
    table.font.size = px(13),
    data_row.padding = px(6)
  )
## Saving the plot
## gtsave(pt, "model_performance.png")

## Heat Map ##

preds <- rr_xgb$prediction()
df_preds <- as.data.table(preds) %>% 
  left_join(data %>% mutate(row_ids = row_number()), by = "row_ids") %>%
  mutate(
    x = 2.5 * (hc_x - 125.42),
    y = 2.5 * (198.27 - hc_y)
  )

## Rough params for a field sketch ##
R_fence <- 400
R_infield <- 140

theta <- seq(-pi/4, pi/4, length.out = 200)

fence_df <- data.frame(
  x = R_fence * sin(theta),
  y = R_fence * cos(theta)
)

infield_df <- data.frame(
  x = R_infield * sin(theta),
  y = R_infield * cos(theta)
)

foul_x <- R_fence * sin(pi/4)
foul_y <- R_fence * cos(pi/4)

## Heat Map Plot ##
p_hit_hm <- ggplot() +
  
  stat_summary_2d(
    data = df_preds,
    aes(x = x, y = y, z = prob.1),
    fun = mean,
    bins = 60
  ) +
  
  scale_fill_viridis_c(
    name = "Hit Probability",
    limits = c(0, 1)
  ) +
  
  ## FOUL LINES ##
  geom_segment(aes(x = 0, y = 0, xend =  foul_x, yend = foul_y),
               color = "white", linewidth = 1) +
  geom_segment(aes(x = 0, y = 0, xend = -foul_x, yend = foul_y),
               color = "white", linewidth = 1) +
  
  ## OUTFIELD FENCE ##
  geom_path(data = fence_df,
            aes(x = x, y = y),
            color = "white",
            linewidth = 1) +
  
  ## INFIELD ARC ##
  geom_path(data = infield_df,
            aes(x = x, y = y),
            color = "white",
            linewidth = 1) +
  
  ## HOME PLATE ##
  geom_point(aes(x = 0, y = 0), color = "white", size = 2) +
  
  coord_fixed(xlim = c(-300, 300), ylim = c(-50, 500)) +
  
  theme_void() +
  theme(
    panel.background = element_rect(fill = "#1f4d2b"),
    plot.background = element_rect(fill = "#1f4d2b"),
    legend.background = element_rect(fill = "#1f4d2b"),
    legend.text = element_text(color = "white"),
    legend.title = element_text(color = "white")
  )
## Saving the plot
## ggsave("plot_heatmap.png", plot=p_hit_hm)

## Feature Importance ##
## SHAP Visualization ##
## SHAP needs 1 model, so train on entire dataset, learn from it ##

learner_xgb$train(task)
xgb_model <- learner_xgb$model

X <- as.matrix(task$data(cols = task$feature_names))
sv <- shapviz(
  xgb_model,
  X_pred = X
)

sv_importance(sv)

## Enhancing the RF Feature Importance viz ##

imp_df <- data.frame(
  Feature = names(imp_avg),
  Importance = as.numeric(imp_avg)
) %>%
  arrange(desc(Importance))

pt_imp <- imp_df %>%
  gt() %>%
  tab_header(
    title = md("**Average Feature Importance (Random Forest)**"),
    subtitle = "Averaged across resampling folds"
  ) %>%
  fmt_number(
    columns = Importance,
    decimals = 3
  ) %>%
  data_color(
    columns = Importance,
    colors = scales::col_numeric(
      palette = c("#f7fbff", "#6baed6", "#08306b"),
      domain = NULL
    )
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      rows = Importance == max(Importance, na.rm = TRUE)
    )
  ) %>%
  cols_align(
    align = "center",
    Importance
  ) %>%
  opt_row_striping()

## Saving the plot
## gtsave(pt_imp, "RF_Feature_Imp.png")

## Calibration Visuals ##
## Not spending much time on these, throwing in the report ##
xgb_calib_viz <- xgb_calib %>%
  gt() %>% 
  tab_header(
    title = md("**Model Calibration -- XGBoost**")
  )
# gtsave(xgb_calib_viz, "XGB_Calib.png")

rf_calib_viz <- rf_calib %>%
  gt() %>% 
  tab_header(
    title = md("**Model Calibration -- Random Forest**")
  )
  
## gtsave(rf_calib_viz, "RF_Calib.png")

gam_calib_viz <- gam_calib %>%
  gt() %>% 
  tab_header(
    title = md("**Model Calibration -- Generalized Additive Model**")
  )
## gtsave(gam_calib_viz, "GAM_Calib.png")

glm_calib_viz <- glm_calib %>%
  gt() %>% 
  tab_header(
    title = md("**Model Calibration -- Logistic Regression**")
  )
## gtsave(glm_calib_viz, "GLM_Calib.png")

svm_calib_viz <- svm_calib %>%
  gt() %>% 
  tab_header(
    title = md("**Model Calibration -- Support Vector Machine**")
  )
## gtsave(svm_calib_viz, "SVM_Calib.png")

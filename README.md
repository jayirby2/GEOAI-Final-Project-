# MLB Hit Probability Model -- Final Project for GIS4123C at the University of Florida 

## Overview
This project models the probability that a ball in play (BIP) results in a hit using MLB Statcast data. By leveraging batted ball characteristics and spatial location, the model captures how contact quality and field positioning influence outcomes.

The project emphasizes **spatial modeling, model comparison, and interpretability** using modern machine learning techniques.

---

## Objectives
- Predict whether a ball in play is a hit or an out
- Compare multiple machine learning models under spatial cross-validation
- Evaluate model calibration and performance
- Visualize hit probability across the baseball field
- Interpret model behavior using feature importance and SHAP values

---

## Data
Data was sourced from Baseball Savant (Statcast) and includes:

- `hc_x`, `hc_y` — batted ball field coordinates  
- `launch_speed` — exit velocity (mph)  
- `launch_angle` — launch angle (degrees)  
- `if_fielding_alignment` — infield positioning  
- `of_fielding_alignment` — outfield positioning  
- `outcome` — hit (1) or out (0)  

---

## Methods

### Models
- XGBoost
- Random Forest
- Generalized Additive Model (GAM)
- Logistic Regression
- Support Vector Machine (SVM)

### Validation
- **Spatial cross-validation (block CV)** to prevent spatial leakage  
- Comparison against standard random CV (baseline)

### Evaluation Metrics
- AUC (Area Under ROC Curve)
- Log Loss
- Brier Score
- Precision / Recall

---

## Key Features

### Spatial Modeling
Field coordinates are transformed into a baseball field representation, allowing for spatial visualization of hit probability.

### Calibration Analysis
Model predictions are evaluated against observed outcomes to ensure probabilistic accuracy.

### Feature Importance
- Random Forest permutation importance  
- SHAP values (XGBoost) for model interpretability  

---

## Results

### Model Performance
Models were evaluated using out-of-sample predictions. XGBoost and Random Forest performed best overall, with strong discrimination and calibration.

### Key Insight
> Depth of contact (`hc_y`) is the dominant predictor of hit probability, while horizontal direction (`hc_x`) plays a secondary role.

This reflects the fundamental importance of **distance over spray angle** in determining outcomes.

---

## Visualizations

### Hit Probability Heatmap
A spatial heatmap showing the probability of a hit across the field:

![Heatmap](output/plot_heatmap.png)

---

### Model Comparison Table
![Model Performance](output/model_performance.png)

---

### Feature Importance
![RF Importance](output/RF_Feature_Imp.png)

---

## Reproducibility

To run the full pipeline:

```r
source("main.R")

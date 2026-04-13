## Data manipulation package ## 
library(tidyverse)

## Data downloaded from -> baseballsavant.com ## 
data <- read.csv("savant_data.csv")

## Select the predictors we want
## Goal: Predict whether ball in play is a hit or an out based on where it 
## is hit on the field and other descriptive contact variables.
## hc_x: x-coordinate of where the ball was fielded (horizontal field location)
## hc_y: y-coordinate of where the ball was fielded (depth on field)
## launch_speed: exit velocity of the ball off the bat (mph)
## launch_angle: vertical angle of the ball off the bat (degrees)
## if_fielding_alignment: infield defensive positioning (e.g., standard, shift)
## of_fielding_alignment: outfield defensive positioning
## outcome: binary result (1 = hit, 0 = out)

data <- data %>%
  select(hc_x, hc_y, launch_speed, launch_angle, 
         if_fielding_alignment, of_fielding_alignment, events) %>%
  mutate(
    outcome = factor(ifelse(
      events %in% c("single", "double", "triple", "home_run"), 1, 0
    ), levels = c(0, 1)),
    
    if_fielding_alignment = as.factor(if_fielding_alignment),
    of_fielding_alignment = as.factor(of_fielding_alignment)
  ) %>%
  select(-events) %>%
  drop_na()

## Data frame should be ready for modeling ##



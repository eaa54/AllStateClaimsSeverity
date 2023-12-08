library(vroom)
library(tidyverse)
library(tidymodels)
library(DataExplorer)

train <- vroom("C:/Users/eaa54/Documents/School/STAT348/AllState/train.csv")
test <- vroom("C:/Users/eaa54/Documents/School/STAT348/AllState/test.csv")

hist(train$loss) #right skewed
plot_correlation(train, type = "continuous") #high correlation off the diagonal

train$loss <- (train$loss + 1)*.25 #based on suggestion from competition winner

allstate_recipe <- recipe(loss ~ ., train) %>%
  update_role(id, new_role = "ID") %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.6) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

##Neural Net##
nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>% #or 100 or 250
  set_engine("nnet") %>% 
  set_mode("classification")

# set workflow
nn_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 75)),
                            levels=3)

# Set up k-fold cross validation and run it
nn_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_nn_results <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

CV_nn_results %>% 
  collect_metrics() %>% 
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

# Find Best Tuning Parameters
bestTune_nn <- CV_nn_results %>%
  select_best("accuracy")

#finalize workflow and fit it
final_nn_wf <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(train)

pred_nn <- predict(final_nn_wf, new_data = test, type = "class") %>%
  bind_cols(., test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_nn, "GGG_preds_nn.csv", delim = ",")

  


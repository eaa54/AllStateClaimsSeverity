library(vroom)
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(embed)
library(bonsai)
library(lightgbm)

train <- vroom("C:/Users/eaa54/Documents/School/STAT348/AllState/train.csv")
test <- vroom("C:/Users/eaa54/Documents/School/STAT348/AllState/test.csv")

#exploratory data analysis
hist(train$loss) #response heavily right skewed
plot_correlation(train, type = "continuous") #some high correlation off the diagonal

train$loss <- (train$loss + 1)^.25 #based on suggestion from competition winner
hist(train$loss) #makes response more normal, but keeps slight right skew

#create recipe
allstate_recipe <- recipe(loss ~ ., train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))

#################################
##PENALIZED LOGISTIC REGRESSION##
#################################
#create model
preg_model <- linear_reg(penalty=tune(), mixture=0) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(preg_model)

#set workflow
preg_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>% #add recipe
  add_model(preg_model) #add model

#set up grid of tuning values
tuning_grid <- grid_regular(penalty(),
                            levels = 5) #levels^2 total tuning possibilities

#set up cv
folds <- vfold_cv(train, v = 5, repeats = 1)


CV_results <- preg_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae)) #mean absolute error used in this kaggle comp

#find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("mae")

#finalize the workflow & fit it
final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

rf_predictions <- final_wf %>%
  predict(new_data = test)

#format for Kaggle
rf_final <- rf_predictions %>%
  bind_cols(test) %>%
  select(id, .pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = (loss)^4-1)

write_csv(rf_final, "PRegSubmission.csv")

#SCORE:1215.35

############
##BOOSTING##
############
#create model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune(),
                          mode = "regression") %>%
               set_engine("lightgbm") #or "xgboost" but lightgbm is faster

#set workflow
boost_wf <- workflow() %>%
  add_recipe(allstate_recipe) %>%
  add_model(boost_model)

#set up tuning grid
boost_tuneGrid <- grid_regular(tree_depth(), trees(), learn_rate(), levels = 3)

#set up cv
boost_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_boost_results <- boost_wf %>%
  tune_grid(resamples = boost_folds,
            grid = boost_tuneGrid,
            metrics = metric_set(mae))

#find best tuning parameters
bestTune_boost <- CV_boost_results %>%
  select_best("mae") #mean absolute error used by this particular Kaggle comp

#finalize workflow and fit it
final_boost_wf <- boost_wf %>%
  finalize_workflow(bestTune_boost) %>%
  fit(train)

#format for kaggle
pred_boost <- predict(final_boost_wf, new_data = test) 
boost_final <- pred_boost %>%
  bind_cols(test) %>%
  select(id,.pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = loss^4-1)

write_csv(boost_final, "boostSubmission.csv")

##SCORE: 1,123.17

#################
##RANDOM FOREST##
#################
#set up the model
my_mod <- rand_forest(mtry = 5,
                      min_n = 2,
                      trees = 1000) %>%
          set_engine("ranger") %>%
          set_mode("regression") #parameters chosen based on classmate's CV (took 20+ hours to run, so for simplicity I used his parameters)

#set up workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(my_mod) %>%
  fit(data = train)

#set up tuning grid - this is how I would tune it
# tuning_grid <- grid_regular(mtry(c(1,5)),
#                             min_n(),
#                             levels = 5) #levels^2 total tuning possibilities
#
# folds <- vfold_cv(train, v = 3, repeats=1)
#
#run the cv
# CV_results <- rf_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#                  metrics = metric_set(mae))
#
#find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best("mae")
#
#finalize the Workflow & fit it
# final_wf <- rf_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = train)

rf_predictions <- rf_wf %>%
  predict(new_data = test)

rf_final <- rf_predictions %>%
  bind_cols(test) %>%
  select(id,.pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = loss^4-1)

write_csv(rf_final, "RFSubmission.csv")

##SCORE: 1,186.36
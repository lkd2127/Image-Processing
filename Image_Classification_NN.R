# Constants
hidden = list(c(20,20), c(50,50), c(70,70), c(100,100,100),
              c(30,30,30), c(50,50,50), c(70,70,70), c(25,25,25,25))
l1= seq(0, 1e-4, 1e-6)
l2= seq(0, 1e-4, 1e-6)
activation = c('Rectifier', 'Tanh', 'Maxout', 
               'RectifierWithDropout', 'TanhWithDropout',
               'MaxoutWithDropout')
max_runtime_secs = 360
max_models = 100
seed = 1031
stopping_rounds_search = 5
stopping_tolerance = 1e-2
pixel = 2:50
label = 1
epochs = 10
stopping_metric = 'logloss'
stopping_rounds_grid = 2

# Evironment set up -  Initialize the H2O cluster and use all available CPUs for parallel processing
h2o.init(nthreads = -1) 

# Functions
## Function to prepare data for H2O
data_input_h2o <- function(x) {
  x$label <- as.factor(x$label) 
  train_h2o = as.h2o(x)
  return(train_h2o)
}

## Function to apply data_input_h2o to multiple datasets
data_input_multiple_h2o <- function(datasets) {
  h2o_datasets <- lapply(datasets, data_input_h2o)
  return(h2o_datasets)
}

## Function to get the best model parameters
get_model_parameters <- function(grid, model_index) {
  model_id <- grid@model_ids[[model_index]]
  model <- h2o.getModel(model_id)
  return(model@parameters)
}

# Data preparation 
h2o_train_datasets <- data_input_multiple_h2o(datasets)
h2o_test_data <- data_input_h2o(data_test)


# Hyperparameters Optimization - Using random seach to automatically tune Hyperparameters
hyperparameters_optimization <- function(grid_id, train_data, validation_data) {
  hyper_parameters = list(activation = activation,
                          hidden = hidden,
                          l1= l1,
                          l2= l2)
  
  search_criteria = list(strategy = 'RandomDiscrete',
                         max_runtime_secs = max_runtime_secs,
                         max_models = max_models,
                         seed = seed,
                         stopping_rounds = stopping_rounds_search,
                         stopping_tolerance = stopping_tolerance)
  
  grid = h2o.grid(algorithm = 'deeplearning',
                  grid_id = "grid_id",
                  training_frame = train_data,
                  validation_frame = validation_data,
                  x = pixel,
                  y = label,
                  epochs = epochs,
                  stopping_metric = stopping_metric,
                  stopping_tolerance = stopping_tolerance,
                  stopping_rounds = stopping_rounds_grid,
                  hyper_params = hyper_parameters,
                  search_criteria = search_criteria)
  
  grid = h2o.getGrid(grid_id = "grid_id",
                     sort_by = stopping_metric,
                     decreasing = FALSE)
  
  return(grid)
}

grid <- hyperparameters_optimization(dat_2000, h2o_train_datasets[[7]], h2o_train_datasets[[8]])


# Fit model
run_nn_model <- function(model_parameters, train_datasets, test_data) {
  results <- list()
  
  for (i in 1:length(train_datasets)) {
    train_data <- train_datasets[[i]]
    sample_size <- nrow(train_data)
    
    start_time <- Sys.time()
    model <- h2o.deeplearning(
      x = pixel,  
      y = label,     
      training_frame = train_data,
      activation = model_parameters$activation,
      hidden = model_parameters$hidden,
      l1 = model_parameters$l1,
      l2 = model_parameters$l2,
      epochs = epochs,
      stopping_metric = stopping_metric,
      stopping_tolerance = stopping_tolerance,
      stopping_rounds = stopping_rounds_grid
    )
    end_time <- Sys.time()
    training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    pred <- h2o.predict(model, test_data)
    end_time_pred <- Sys.time()
    prediction_time <- as.numeric(difftime(end_time_pred, end_time, units = "secs"))
    
    total_runtime <- training_time + prediction_time
    
    results[[i]] <- list(
      model_id = model@model_id,
      prediction = pred$predict,
      sample_size = sample_size,     
      total_runtime = total_runtime
    )
  }
  
  return(results)
}

# Model results
## Model 1
### Get the best performing model's parameters
mod_nn_1_parameters <- get_model_parameters(grid, 1)

### Fit model 
model_nn_1_results <- run_nn_model(mod_nn_1_parameters, h2o_train_datasets, h2o_test_data)

### Model Scoring 
mod_nn_1_preliminary_scores <- calculate_score_table(model_nn_1_results, 
                                                     total_training_rows, 
                                                     h2o_test_data)
datatable(mod_nn_1_preliminary_scores)

### Extract the maximum and minimum points along with their corresponding sample sizes
mod_nn_1_max_min_points <- extract_max_min_points(mod_nn_1_preliminary_scores)

## Model 2
### Get the 2nd best performing model's parameters
mod_nn_2_parameters <- get_model_parameters(grid, 2)

##3 Save the mod_nn_2_parameters object as an RDS file for models accuracy analysis 
saveRDS(mod_nn_2_parameters, "~/Desktop/mod_nn_2_parameters.rds")

### Fit model 
model_nn_2_results <- run_nn_model(mod_nn_2_parameters, h2o_train_datasets, h2o_test_data)

### Model Scoring 
mod_nn_2_preliminary_scores <- calculate_score_table(model_nn_2_results, 
                                                     total_training_rows, 
                                                     h2o_test_data)
datatable(mod_nn_2_preliminary_scores)

### Extract the maximum and minimum points along with their corresponding sample sizes
mod_nn_2_max_min_points <- extract_max_min_points(mod_nn_2_preliminary_scores)

## Model 3
### Get the 3rd best performing model's parameters
mod_nn_3_parameters <- get_model_parameters(grid, 3)

### Fit model 
model_nn_3_results <- run_nn_model(mod_nn_3_parameters, h2o_train_datasets, h2o_test_data)

### Model Scoring 
mod_nn_3_preliminary_scores <- calculate_score_table(model_nn_3_results, 
                                                     total_training_rows, 
                                                     h2o_test_data)
datatable(mod_nn_3_preliminary_scores)

#### Extract the maximum and minimum points along with their corresponding sample sizes
mod_nn_3_max_min_points <- extract_max_min_points(mod_nn_3_preliminary_scores)





module SModels

  # dependencies
  using DecisionTree
  using ScikitLearn
  using ScikitLearn.GridSearch: GridSearchCV
  using ScikitLearn.CrossValidation: cross_val_predict
  using PyCall
  using PyCallJLD
  using JLD
  using DistributedArrays
  using Distributions

  # imports from ScikitLearn
  @sk_import neural_network: MLPRegressor  #to train the regressor
  @sk_import neural_network: MLPClassifier #to train the classifier (convergence or not of the model)
  @sk_import preprocessing: StandardScaler #to scale observations
  @sk_import metrics: accuracy_score       #accuracy of classifier

  # package code goes here
  include("types.jl")
  include("generic.jl")
  include("draws.jl")
  include("train.jl")


  # types.jl
  #---------
  export SModelsProblem, SModelsOptions, default_function

  #generic.jl
  #-----------
  export split_train_test, convert_to_array_of_arrays, convert_to_array
  export predict_scaled_robust
  export convert_vector_to_array, convert_vector_to_array, predict_one_obs_scaled
  export predict_one_obs_unscaled, predict_one_obs_unscaled, set_model_function!
  export calculate_mean_per_error, calculate_median_abs_per_error
  export calculate_maximum_abs_per_error
  export set_bounds!
  export date_now

  # draws.jl
  #---------
  export create_grid_stochastic, generate_std
  # train.jl
  #---------
  export train_surrogate_model
  export evaluateModel!

end # module

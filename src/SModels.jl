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
  using LatinHypercubeSampling
  using Distributions

  # imports from ScikitLearn
  @sk_import neural_network: MLPRegressor
  @sk_import preprocessing: StandardScaler

  # package code goes here
  include("types.jl")
  include("generic.jl")
  include("draws.jl")
  include("train.jl")


  # types.jl
  #---------
  export SModelsProblem, SModelsOptions

  #generic.jl
  #-----------
  export split_train_test, convert_to_array_of_arrays, convert_to_array
  export convert_vector_to_array, convert_vector_to_array, predict_one_obs_scaled
  export predict_one_obs_unscaled, predict_one_obs_unscaled, set_model_function!
  export calculate_mean_per_error, calculate_max_per_error, calculate_mean_per_error
  export calculate_mean_per_error, calculate_maximum_per_error, calculate_maximum_per_error
  export set_bounds!

  # draws.jl
  #---------
  export create_grid_stochastic, generate_std
  # train.jl
  #---------
  export train_sModel
  export evaluateModel!

end # module

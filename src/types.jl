
"""
	SModelsOptions
SModelsOptions is a struct that contains options necessary to train the surrogate
model
"""
struct SModelsOptions
  sModelType::Symbol          # The type of model to use for the surrogate model
  desiredMaxPerError::Float64 # The desired percentage error after which training is stopped
	desiredMinObs::Int64			# Minimum number of points in the train sample before stopping
  nWorkers::Int64          # To store the number of workers
  nbBatches::Int64         # Total number of batches before stopping
  batchSizeWorker::Int64   # Number of evals done by each worker at each round (multiple of nworkers())
  batchSize::Int64         # batchSize = nworkers()*batchSizeWorker
  maxEvals::Int64          # maxEvals = nbBatches*batchSize
  testTrainRatio::Float64  # size of the test sample relative to the train one
  gridType::Symbol         # Type of grid used when sampling from the parameter space
end

function SModelsOptions( ; sModelType = :MLPRegressor,
                          desiredMaxPerError::Float64 = 0.5,
													desiredMinObs::Int64 = 1000, # The desired percentage error after which training is stopped
                          nWorkers::Int64 = nworkers(),          # To store the number of workers
                          nbBatches::Int64 = 10,        # Total number of batches before stopping
                          batchSizeWorker::Int64 = 10,  # Number of evals done by each worker at each round (multiple of nworkers())
                          batchSize::Int64 = nworkers()*batchSizeWorker,         # batchSize = nworkers()*batchSizeWorker
                          maxEvals::Int64 = nbBatches*batchSize,        # maxEvals = nbBatches*batchSize
                          testTrainRatio::Float64 = 0.25,  # size of the test sample relative to the train one
                          gridType::Symbol = :uniform)         # Type of grid used when sampling from the parameter space


    SModelsOptions(sModelType,
                  desiredMaxPerError,
									desiredMinObs,
                  nWorkers,
                  nbBatches,
                  batchSizeWorker,
                  batchSize,
                  maxEvals,
                  testTrainRatio,
                  gridType)
end
"""
	SModelsProblem
SModelsProblem is a mutable struct that caries all the information needed to
build a surrogate model.
"""
mutable struct SModelsProblem
  modelFunction::Function      #function f:x -> y that we are trying to approximate
  lowerBound::Array{Float64,1} #lower bound for the parameter space
  upperBound::Array{Float64,1} #upper bound for the parameter space
  dimX::Int64                  #dimension of the input parameter
  dimY::Int64                  #dimension of the output vector
  options::SModelsOptions      #options
  trainingSuccessful::Bool     #desiredPerError reached?
  scaler::PyObject             #to rescale the data
end


function SModelsProblem( ;modelFunction::Function = default_function,    #function f:x -> y that we are trying to approximate
                          lowerBound::Array{Float64,1} = zeros(1), #lower bound for the parameter space
                          upperBound::Array{Float64,1} = ones(1),                   #upper bound for the parameter space
                          dimX::Int64 = 1,                #dimension of the input parameter
                          dimY::Int64 = 1,                 #dimension of the output vector
                          options::SModelsOptions = SModelsOptions(),     #options
                          trainingSuccessful::Bool = false,     #desiredPerError reached?
                          scaler::PyObject = StandardScaler() )


  SModelsProblem(modelFunction,      #function f:x -> y that we are trying to approximate
                lowerBound, #lower bound for the parameter space
                upperBound, #upper bound for the parameter space
                dimX,                #dimension of the input parameter
                dimY,                 #dimension of the output vector
                options,      #options
                trainingSuccessful,     #desiredPerError reached?
                scaler)
end


"""
	default_function(x)
Function x->x. Used to initialize functions.
"""
function default_function(x)
	println("default_function, returns input")
	return x
end

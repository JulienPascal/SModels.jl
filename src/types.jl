
"""
	SModelsOptions
SModelsOptions is a struct that contains options necessary to train the surrogate
model
"""
struct SModelsOptions
  sModelType::Symbol       # The type of model to use for the surrogate model
  nbDraws::Int64           # Number of draws from the parameter space
  desiredPerError::Float64 # The desired percentage error after which training is stopped
  nWorkers::Int64          # To store the number of workers
  nbBatches::Int64         # Total number of batches before stopping
  batchSizeWorker::Int64   # Number of evals done by each worker at each round (multiple of nworkers())
  batchSize::Int64         # batchSize = nworkers()*batchSizeWorker
  maxEvals::Int64          # maxEvals = nbBatches*batchSize
  onCluster::Bool          # Are you on a cluster
  clusterType::Symbol      # Type of cluster used
  testTrainRatio::Float64  # size of the test sample relative to the train one
end

"""
	SModelsProblem
SModelsProblem is a mutable struct that caries all the information needed to
build a surrogate model.
"""
mutable struct SMMProblem
  modelFunction::Function      #function f:x -> y that we are trying to approximate
  lowerBound::Array{Float64,1} #lower bound for the parameter space
  upperBound::Array{Float64,1} #upper bound for the parameter space
  dimX::Int64                  #dimension of the input parameter
  dimY::Int64                  #dimension of the output vector
  options::SModelsOptions      #options
  trainingSuccessful::Bool     #desiredPerError reached?
end


"""
	default_function(x)
Function x->x. Used to initialize functions.
"""
function default_function(x)
	println("default_function, returns input")
	x
end

# * use DistributedArrays
# *

function train_sModel(sModelsProblem::SModelsProblem; verbose::Bool=false, saveToDisk::Bool=false)

  #initialization
  clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes = (2), random_state=1)

  if verbose == true
    info("nworkers = $(nworkers())")
  end

  # Initialization
  XDense = Array{Float64}(0,0)
  YDense = Array{Float64}(0,0)

  # Generate random draws from the parameter space
  listGrids = []
  for i=1:sModelsProblem.options.nbBatches
    push!(listGrids, create_grid_stochastic(sModelsProblem.lowerBound, sModelsProblem.upperBound, sModelsProblem.options.batchSize, gridType = sModelsProblem.options.gridType))
  end

  counterBatches = 0

  for i=1:sModelsProblem.options.nbBatches

    counterBatches+=1

    if verbose == true
      info("nbBatches = $(i)")
      info("nb function evaluations = $(i*sModelsProblem.options.batchSize)")
    end

    # row dimension = moving along observations
    # column dimension = moving along one observation
    # Model's input:
    #---------------
    # not going to be modified
    #-------------------------
    XDistributed = distribute(listGrids[i], procs = workers(), dist = [nworkers(), 1])

    # Model's output
    #---------------
    # is going to be modified
    #-------------------------
    YDistributed = distribute(zeros(sModelsProblem.options.batchSize, sModelsProblem.dimY),  procs = workers(), dist = [nworkers(), 1])

    # Split the work on the cluster
    #------------------------------
    refs = Array{Any,1}(nworkers())
    @sync for (wIndex, wID) in enumerate(workers())
      # Each worker modifies its part of XDistributed and YDistributed
      @async refs[wIndex] = @spawnat wID evaluateModel!(sModelsProblem, localpart(XDistributed), localpart(YDistributed))
    end

    # synchronization
    #----------------
    for (wIndex, wID) in enumerate(workers())
      while isready(refs[wIndex]) == false end
    end


    # Concatenate results
    #---------------------
    # First iteration of the loop is a special case
    if XDense == Array{Float64}(0,0)
      XDense = listGrids[i]
    else
      XDense = vcat(XDense, listGrids[i])
    end

    if YDense == Array{Float64}(0,0)
      YDense = convert(Array, YDistributed)
    else
      YDense = vcat(YDense, convert(Array, YDistributed))
    end

    # Split between train and test samples
    #--------------------------------------
    XTrain, XTest, yTest, yTrain = split_train_test(XDense, YDense, testRatio = false)

    # Scale the data
    #---------------
    sModelsProblem.scaler = StandardScaler()
    fit!(sModelsProblem.scaler, XTrain)
    XTrainScaled = transform(sModelsProblem.scaler, XTrain)
    XTestScaled = transform(sModelsProblem.scaler, XTest)


    # Train the model, use cross-validation
    #-------------------------------------
    # TODO
    if verbose == true
      info("Training the model")
    end

    # There is certainly a better way of doing that
    d = Dict(:hidden_layer_sizes => ((10), (20), (30), (40), (10, 10), (10, 20), (10, 30), (20, 10), (20, 20), (20, 30), (30, 10), (30, 20), (30, 30)))
    bestMaxPerError = Inf
    bestParam = d[:hidden_layer_sizes][1]

    # Loop over hidden layers configuration
    # Looking fot the hidden layer configuration that minimizes the max error in the test set
    for (kIndex, kValue) in enumerate(d[:hidden_layer_sizes])

        println(kValue)
        clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes = kValue, random_state=1)
        fit!(clfr, XTrainScaled, yTrain)

        yPredicted = predict(clfr, XTestScaled)
        maxPerErr = calculate_maximum_abs_per_error(yTest, yPredicted)

        if maxPerErr < bestMaxPerError
            bestMaxPerError = maxPerErr
            bestParam = kValue
        end

    end

    if verbose == true
      info("Best hyper-parameters: $(bestParam)")
    end

    # training network using the "best" hidden layer
    #-----------------------------------------------
    clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=bestParam, random_state=1)
    fit!(clfr, XTrainScaled, yTrain)

    # Train set
    #----------
    yPredicted = predict(clfr, XTrainScaled)

    if verbose == true
      info("Mean Percentage Error Train Set = $(calculate_mean_per_error(yTrain, yPredicted))")
      info("Abs Maximum Percentage Error Train Set = $(calculate_maximum_abs_per_error(yTrain, yPredicted))")
    end

    # Test set
    #---------
    yPredicted = predict(clfr, XTestScaled)

    if verbose == true
      info("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
      info("Abs Maximum Percentage Error Test Set = $(calculate_maximum_abs_per_error(yTest, yPredicted))")
    end

    # If requested, save the model
    #-----------------------------
    if saveToDisk == true

      if verbose == true
        info("Saving files to disk")
      end

      JLD.save("clfr_$(date_now()).jld", "clfr", clfr)
      JLD.save("XDense_$(date_now()).jld", "XDense", XDense)
      JLD.save("YDense_$(date_now()).jld", "YDense", YDense)

    end
    # Test model accuracy
    # If good enough, stop and save the model
    # Else, move the next batch
    #----------------------------------------
    if calculate_maximum_abs_per_error(yTest, yPredicted) < sModelsProblem.options.desiredMaxPerError

      if verbose == true
        info("Desired max percentage error = $(sModelsProblem.options.desiredMaxPerError)")
        info("Size of the train sample = $(size(XTrain,1))")
      end

      if size(XTrain,1) > sModelsProblem.options.desiredMinObs

        if verbose == true
          info("Minimum number of points in the train sample reaching. Training succesfull.")
        end

        sModelsProblem.trainingSuccessful = true
        break

      end

    end

  end # for i=1:nbBatches

  # If training was not succesfull
  #-------------------------------
  if counterBatches == sModelsProblem.options.nbBatches
    info("Max number of iterations reached. Training failure.")
    sModelsProblem.trainingSuccessful = false
  end

  return clfr

end


function evaluateModel!(sModelsProblem::SModelsProblem, XDistributed::Array, YDistributed::Array; penaltyValue::Float64 = 9999999.0)

  # Loop over local part of XDistributed
  #-------------------------------------
  for i=1:size(XDistributed, 1)

    try
      YDistributed[i,:] = sModelsProblem.modelFunction(XDistributed[i,:])
    catch errF
      info("$(errF)")
      YDistributed[i,:] = ones(sModelsProblem.dimX)*penaltyValue
    end

  end

end

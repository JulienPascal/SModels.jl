# * use DistributedArrays
# *

function train_sModel(sModelsProblem::SModelsProblem; verbose::Bool=false, saveToDisk::Bool=false, robust::Bool=true)

  starting_date = date_now()

  #initialization:
  #--------------
  # regression model to predict the continuous output:
  clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes = (2), random_state=1)
  # classifier to predict convergence of the model:
  clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes = (2), random_state=1)
  # maximum absolute value percentage error regressor:
  max_abs_per_regressor = Inf
  # accuracy score for the classifier:
  accuracy_classifier = Inf
  # size of the train sample:
  size_trainSample = 0

  if verbose == true
    info("Starting date = $(starting_date)")
    info("nworkers = $(nworkers())")
  end

  # Initialization objects on process 1 only:
  #-----------------------------------------
  # To store parameter values
  XDense = Array{Float64}(0,0)
  # To store output values
  YDense = Array{Float64}(0,0)
  # To store convergence status:
  YConvergenceDense = Array{Int64}(0)

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

    # Objects distributed on the cluster
    #-----------------------------------
    # row dimension = moving along observations
    # column dimension = moving along one observation
    # Model's input:
    #---------------
    # not going to be modified
    #-------------------------
    XDistributed = distribute(listGrids[i], procs = workers(), dist = [nworkers(), 1])

    # Model's output
    #---------------
    # are going to be modified
    #-------------------------
    YDistributed = distribute(zeros(sModelsProblem.options.batchSize, sModelsProblem.dimY),  procs = workers(), dist = [nworkers(), 1])
    YConvergenceDistributed = distribute(zeros(Int64, sModelsProblem.options.batchSize),  procs = workers(), dist = [nworkers()])

    # Split the work on the cluster
    #------------------------------
    refs = Array{Any,1}(nworkers())
    @sync for (wIndex, wID) in enumerate(workers())
      # Each worker modifies its part of XDistributed and YDistributed
      @async refs[wIndex] = @spawnat wID evaluateModel!(sModelsProblem, localpart(XDistributed), localpart(YDistributed), localpart(YConvergenceDistributed))
    end

    # synchronization barrier
    #------------------------
    for (wIndex, wID) in enumerate(workers())
      while isready(refs[wIndex]) == false
        if verbose == true
          info("Waiting for worker $(wID).")
        end
      end
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

    if YConvergenceDense == Array{Int64}(0)
      YConvergenceDense = convert(Array, YConvergenceDistributed)
    else
      YConvergenceDense = vcat(YConvergenceDense, convert(Array, YConvergenceDistributed))
    end

    # Split between train and test samples
    #--------------------------------------
    XTrain, XTest, yTrain, yTest, yConvergenceTrain, yConvergenceTest = split_train_test(XDense, YDense, YConvergenceDense, testRatio = false)

    # If robust is true, train the regression on parameter values for which
    # convergence was reached
    #---------------------------------------------------------------------------
    XTrainRobust = XTrain[yConvergenceTrain .== sModelsProblem.options.convergenceFlag]
    XTestRobust = XTest[yConvergenceTest .== sModelsProblem.options.convergenceFlag]
    yTrainRobust = yTrain[yConvergenceTrain .== sModelsProblem.options.convergenceFlag]
    yTestRobust = yTest[yConvergenceTest .== sModelsProblem.options.convergenceFlag]

    if robust == false
      size_trainSample = size(XTrain, 1)
    else
      size_trainSample = size(XTrainRobust, 1)
    end


    # Scale the data
    #---------------
    sModelsProblem.scaler = StandardScaler()
    sModelsProblem.scalerRobust = StandardScaler()

    fit!(sModelsProblem.scaler, XTrain)
    fit!(sModelsProblem.scalerRobust, XTrainRobust)

    XTrainScaled = transform(sModelsProblem.scaler, XTrain)
    XTestScaled = transform(sModelsProblem.scaler, XTest)

    XTrainScaledRobust =  transform(sModelsProblem.scalerRobust, XTrainRobust)
    XTestScaledRobust = transform(sModelsProblem.scalerRobust, XTestRobust)

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # I.Train another model to predict convergence versus not convergence of the
    # model.
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    if verbose == true
      info("Training the classifier")
    end

    d = Dict(:hidden_layer_sizes => ((10), (20), (30), (40),
                                    (10, 10), (10, 20), (10, 30), (10, 40),
                                    (20, 10), (20, 20), (20, 30), (20, 40),
                                    (30, 10), (30, 20), (30, 30), (30, 40),
                                    (40, 10), (40, 20), (40, 30), (40, 40),
                                    (10,10,10), (20,20,20), (30,30,30), (40,40,40)))
    bestAccuracyScore = 0
    bestParam = d[:hidden_layer_sizes][1]

    # Loop over hidden layers configuration
    # Looking fot the hidden layer configuration that minimizes the max error in the test set
    for (kIndex, kValue) in enumerate(d[:hidden_layer_sizes])

        println(kValue)
        clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes = kValue, random_state=1)
        fit!(clf, XTrainScaled, yConvergenceTrain)

        yPredicted = round(Int64, predict(clf, XTestScaled))
        accuracyScore = accuracy_score(yConvergenceTest, yPredicted)

        # Maximize the accuracy:
        if accuracyScore > bestAccuracyScore
            bestAccuracyScore = accuracyScore
            bestParam = kValue
        end

    end

    if verbose == true
      info("Best hyper-parameters for classifier: $(bestParam)")
    end

    # training classifier using the "best" hidden layer
    #-----------------------------------------------
    clf = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=bestParam, random_state=1)
    fit!(clf, XTrainScaled, yConvergenceTrain)

    # Train set
    #----------
    yPredicted = round(Int64, predict(clf, XTrainScaled))

    if verbose == true
      info("Classifier")
      info("Accuracy Score Train Set = $(accuracy_score(yConvergenceTrain, yPredicted))")

    end

    # Test set
    #---------
    yPredicted = round(Int64, predict(clf, XTestScaled))
    accuracy_classifier = accuracy_score(yConvergenceTest, yPredicted)

    if verbose == true
      info("Classifier")
      info("Accuracy Score Test Set = $(accuracy_classifier)")
    end

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # II. Train the model to predict the continuous output
    # Use "manual" cross-validation on train versus test samples
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    if verbose == true
      info("Training the regression model")
      info("Robust = $(robust)")
    end

    # If robust, train the model only on

    # TODO: automatic way of doing that + adapt to the number of inputs and output
    # Low tech: There is certainly a better way of doing that
    d = Dict(:hidden_layer_sizes => ((10), (20), (30), (40),
                                    (10, 10), (10, 20), (10, 30), (10, 40),
                                    (20, 10), (20, 20), (20, 30), (20, 40),
                                    (30, 10), (30, 20), (30, 30), (30, 40),
                                    (40, 10), (40, 20), (40, 30), (40, 40),
                                    (10,10,10), (20,20,20), (30,30,30), (40,40,40)))
    bestMaxPerError = Inf
    bestParam = d[:hidden_layer_sizes][1]

    # Loop over hidden layers configuration
    # Looking fot the hidden layer configuration that minimizes the max error in the test set
    for (kIndex, kValue) in enumerate(d[:hidden_layer_sizes])

        println(kValue)
        clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes = kValue, random_state=1)

        if robust == false

          fit!(clfr, XTrainScaled, yTrain)
          yPredicted = predict(clfr, XTestScaled)
          maxPerErr = calculate_maximum_abs_per_error(yTest, yPredicted)

        # If robust = true, use only the data for which convergence was reached
        #-----------------------------------------------------------------------
        else

          fit!(clfr, XTrainScaledRobust, yTrainRobust)
          yPredicted = predict(clfr, XTestScaledRobust)
          maxPerErr = calculate_maximum_abs_per_error(yTestRobust, yPredicted)

        end

        if maxPerErr < bestMaxPerError
            bestMaxPerError = maxPerErr
            bestParam = kValue
        end

    end

    if verbose == true
      info("Best hyper-parameters for regression model: $(bestParam)")
    end

    clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=bestParam, random_state=1)
    # training network using the "best" hidden layer
    #-----------------------------------------------
    if robust == false
      # Train Set:
      #-----------
      fit!(clfr, XTrainScaled, yTrain)
      yPredicted = predict(clfr, XTrainScaled) # Train set
      mean_per_regressorTrain = calculate_mean_per_error(yTest, yPredicted)
      max_abs_per_regressorTrain = calculate_maximum_abs_per_error(yTest, yPredicted)
      # Test Set:
      #-----------
      yPredicted = predict(clfr, XTestScaled)
      mean_per_regressor = calculate_mean_per_error(yTest, yPredicted)
      max_abs_per_regressor = calculate_maximum_abs_per_error(yTest, yPredicted)

    # If robust = true, use only the data for which convergence was reached
    #-----------------------------------------------------------------------
    else
      # Train Set:
      #-----------
      fit!(clfr, XTrainScaledRobust, yTrainRobust)
      yPredicted = predict(clfr, XTestScaledRobust) # Train set
      mean_per_regressorTrain = calculate_mean_per_error(yTestRobust, yPredicted)
      max_abs_per_regressorTrain = calculate_maximum_abs_per_error(yTestRobust, yPredicted)
      # Test Set:
      #-----------
      yPredicted = predict(clfr, XTestScaledRobust)
      mean_per_regressor = calculate_mean_per_error(yTestRobust, yPredicted)
      max_abs_per_regressor = calculate_maximum_abs_per_error(yTestRobust, yPredicted)
    end

    if verbose == true
      info("Regressor:")
      info("Train set:")
      info("Mean Percentage Error Train Set = $(mean_per_regressorTrain)")
      info("Maximum Abs Percentage Error Train Set = $(max_abs_per_regressorTrain)")
      info("Test set:")
      info("Mean Percentage Error Test Set = $(mean_per_regressor)")
      info("Maximum Abs Percentage Error Test Set = $(max_abs_per_regressor)")
    end

    # If requested, save the model
    #-----------------------------
    if saveToDisk == true

      if verbose == true
        info("Saving files to disk")
      end

      # regressor:
      JLD.save("clfr_$(starting_date).jld", "clfr", clfr)
      # classifier:
      JLD.save("clf_$(starting_date).jld", "clf", clf)
      # scaler:
      JLD.save("scaler_$(starting_date).jld", "scaler", sModelsProblem.scaler)
      # scaler robust:
      JLD.save("scalerRobust_$(starting_date).jld", "scalerRobust", sModelsProblem.scalerRobust)
      # data:
      JLD.save("XDense_$(starting_date).jld", "XDense", XDense)
      JLD.save("YDense_$(starting_date).jld", "YDense", YDense)
      JLD.save("YConvergenceDense_$(starting_date).jld", "YConvergenceDense", YConvergenceDense)

    end

    #-----------------------------------------
    # Test model accuracy
    # If good enough, stop and save the model
    # Else, move the next batch
    #-----------------------------------------
    # Test on the regressor:
    #-----------------------
    if max_abs_per_regressor < sModelsProblem.options.desiredMaxPerErrorRegressor

      if verbose == true
        info("Current max percentage error regressor = $(max_abs_per_regressor)")
        info("Desired max percentage error regressor = $(sModelsProblem.options.desiredMaxPerErrorRegressor)")
        info("Size of the train sample = $(size(XTrain,1))")
      end

      # Test on the classifier
      #-----------------------
      if accuracy_classifier < sModelsProblem.options.desiredAccuracyClassifier

        if verbose == true
          info("Current accuracy score classifier = $(accuracy_classifier)")
          info("Desired accuracy score classifier = $(sModelsProblem.options.desiredAccuracyClassifier)")
        end

        # Test on sample size:
        #---------------------
        if size_trainSample > sModelsProblem.options.desiredMinObs

          if verbose == true
            info("Minimum number of points in the train sample reached. Training succesfull.")
          end

          sModelsProblem.trainingSuccessful = true
          break

        end

      end

    end

  end # for i=1:nbBatches

  # If training was not succesfull
  #-------------------------------
  if counterBatches == sModelsProblem.options.nbBatches

    if verbose == true
      info("----------------------------------------------------")
      info("Max number of iterations reached. Training failure.")
      info("Size of the train sample = $(size_trainSample)")
      info("Max abs percentage error regressor = $(max_abs_per_regressor)")
      info("Accuracy score classifier = $(accuracy_classifier)")
      info("----------------------------------------------------")
    end

    sModelsProblem.trainingSuccessful = false
  end

  return clfr, clf

end


function evaluateModel!(sModelsProblem::SModelsProblem, XDistributed::Array, YDistributed::Array, YConvergenceDistributed::Array)

  # Loop over local part of XDistributed
  #-------------------------------------
  for i=1:size(XDistributed, 1)

    # set penalty value by default:
    YDistributed[i,:] = ones(sModelsProblem.dimX)*sModelsProblem.options.penaltyValue
    YConvergenceDistributed[i] = sModelsProblem.options.nonConvergenceFlag

    try
      # if model is evaluated successfully, penalty values are overwritten
      #-------------------------------------------------------------------
      YDistributed[i,:] = sModelsProblem.modelFunction(XDistributed[i,:])
      YConvergenceDistributed[i] = sModelsProblem.options.convergenceFlag
    catch errF
      info("$(errF)")
    end

  end

end

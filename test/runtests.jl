using SModels
using ScikitLearn
using ScikitLearn.GridSearch: GridSearchCV
using JLD, PyCallJLD
@sk_import neural_network: MLPClassifier
@sk_import neural_network: MLPRegressor
@sk_import preprocessing: StandardScaler
@sk_import metrics: accuracy_score

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

srand(1234)

# This line should fail:
# @test 1 == 2

# Can we reproduce a basic example?
@testset "Testing Examples ScikitLearn.jl" begin


    X = [[0., 0.], [1., 1.]]
    y = [0, 1]

    clf = MLPClassifier(solver="lbfgs", alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

    fit!(clf, X, y)

    res = predict(clf, [[2., 2.], [-1., -2.]])

    @test res[1] == 1
    @test res[2] == 0


    # Saving and loading
    #-------------------
    JLD.save("clf.jld", "clf", clf)
    clfloaded = JLD.load("clf.jld", "clf")

    res = predict(clfloaded, [[2., 2.], [-1., -2.]])

    @test res[1] == 1
    @test res[2] == 0



end

@testset "Testing generic functions" begin


    # 2d->2d function we aim at fitting
    function trueFunction(x::Float64, y::Float64)
        [x^2 + y^2, exp(x)*cos(y)]
    end

    function trueFunction(z::Array{Float64,1})
        trueFunction(z[1], z[2])
    end

    # Function that fails  when z[2] > 0.5
    # to test if the package is able to handle functions
    # that may fail
    function trueFunctionRisk(z::Array{Float64,1})
        if z[2] < 0.5
            return trueFunction(z[1], z[2])
        else
            error("Failure of trueFunctionRisk.")
        end
    end


    # create an input matrix
    function createX(;dimInput = 2, lowerBoundX = -1.0, lowerBoundY = -1.0, upperBoundX = 1.0, upperBoundY = 1.0, nbPointsX = 10, nbPointsY = 10)

        # initialization
        X = Array{Array{Float64,1},1}(nbPointsX*nbPointsY)

        for (kIndex, tupleValue) in enumerate(collect(Iterators.product(linspace(lowerBoundX, upperBoundX, nbPointsX), linspace(lowerBoundY, upperBoundY, nbPointsY))))
            X[kIndex] = [tupleValue[1], tupleValue[2]]
        end

        return X

    end

    # create output matrix
    function createY(X::Array{Array{Float64,1},1}; dimOutput = 2)

        # initialization
        y = Array{Array{Float64,1},1}(size(X,1))

        # Looping over X
        for xIndex = 1:size(X,1)
            y[xIndex] = trueFunction(X[xIndex][1], X[xIndex][2])
        end

        return y

    end

    # create output matrix
    # penalty value when trueFunctionRisk fails
    function createYRisk(X::Array{Array{Float64,1},1}; dimOutput::Int64 = 2, penaltyValue::Float64 = 999.0)

        # initialization
        y = Array{Array{Float64,1},1}(size(X,1))

        # Looping over X
        for xIndex = 1:size(X,1)
          if X[xIndex][2] < 0.5
            y[xIndex] = trueFunction(X[xIndex][1], X[xIndex][2])
          else
            y[xIndex] = ones(dimOutput)*penaltyValue
          end

        end

        return y

    end


    @testset "Testing fitting a 2d->2d function" begin

        aTolMeanPerError = 0.05
        aTolMaxPerError = 1.0
        aTolScaled = 0.001

        #input
        X = createX()
        #ouptut
        y = createY(X)

        # splitting between train and test samples
        XTrain, XTest, yTrain, yTest = split_train_test(X, y, testRatio = false)

        # scaling
        scaler = StandardScaler()
        fit!(scaler, XTrain)
        XTrainScaled = transform(scaler, XTrain)
        XTestScaled = transform(scaler, XTest)

        # training network
        clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(30, 30), random_state=1)
        fit!(clfr, XTrainScaled, yTrain)

        # Train set
        #----------
        yPredicted = predict(clfr, XTrainScaled)

        println("Mean Percentage Error Train Set = $(calculate_mean_per_error(yTrain, yPredicted))")
        println("Maximum Abs Percentage Error Train Set = $(calculate_maximum_abs_per_error(yTrain, yPredicted))")
        @test abs(calculate_mean_per_error(yTrain, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_abs_per_error(yTrain, yPredicted)) < aTolMaxPerError


        # Test set
        #---------
        yPredicted = predict(clfr, XTestScaled)

        println("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
        println("Maximum Abs Percentage Error Test Set = $(calculate_maximum_abs_per_error(yTest, yPredicted))")
        @test abs(calculate_mean_per_error(yTest, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_abs_per_error(yTest, yPredicted)) < aTolMaxPerError

        yPredictedScaled = predict_one_obs_scaled(clfr, XTrainScaled[1,:])
        yPredictedUnscaled = predict_one_obs_unscaled(clfr, XTrain[1,:], scaler)

        #scaled versus unscaled observation
        @test yPredictedScaled ≈ yPredictedUnscaled atol = aTolScaled

    end

    #cross-validation to choose the hidden-layer
    #-------------------------------------------------
    @testset "Testing fitting a 2d->2d function tuning hyper-parameters" begin

        aTolMeanPerError = 0.05
        aTolMaxPerError = 1.2
        aTolScaled = 0.001

        #input
        X = createX()
        #ouptut
        y = createY(X)

        # splitting between train and test samples
        XTrain, XTest, yTrain, yTest = split_train_test(X, y, testRatio = false)

        # scaling
        scaler = StandardScaler()
        fit!(scaler, XTrain)
        XTrainScaled = transform(scaler, XTrain)
        XTestScaled = transform(scaler, XTest)

        gridsearch = GridSearchCV(MLPRegressor(solver="lbfgs", alpha=1e-5, random_state=1), Dict(:hidden_layer_sizes => ((10), (20), (30), (40), (10, 10), (10, 20), (10, 30), (20, 10), (20, 20), (20, 30), (30, 10), (30, 20), (30, 30))))
        fit!(gridsearch, X, y)
        println("Best hyper-parameters: $(gridsearch.best_params_)")

        # training network using the "best" hidden layer
        #-----------------------------------------------
        clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=gridsearch.best_params_[:hidden_layer_sizes], random_state=1)
        fit!(clfr, XTrainScaled, yTrain)

        # Train set
        #----------
        yPredicted = predict(clfr, XTrainScaled)

        println("Mean Percentage Error Train Set = $(calculate_mean_per_error(yTrain, yPredicted))")
        println("Maximum Abs Percentage Error Train Set = $(calculate_maximum_abs_per_error(yTrain, yPredicted))")
        @test abs(calculate_mean_per_error(yTrain, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_abs_per_error(yTrain, yPredicted)) < aTolMaxPerError


        # Test set
        #---------
        yPredicted = predict(clfr, XTestScaled)

        println("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
        println("Maximum Abs Percentage Error Test Set = $(calculate_maximum_abs_per_error(yTest, yPredicted))")
        @test abs(calculate_mean_per_error(yTest, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_abs_per_error(yTest, yPredicted)) < aTolMaxPerError

        yPredictedScaled = predict_one_obs_scaled(clfr, XTrainScaled[1,:])
        yPredictedUnscaled = predict_one_obs_unscaled(clfr, XTrain[1,:], scaler)

        #scaled versus unscaled observation
        @test yPredictedScaled ≈ yPredictedUnscaled atol = aTolScaled

    end

    #cross-validation to choose the hidden-layer done "manually"
    #-----------------------------------------------------------
    @testset "Testing fitting a 2d->2d function manually tuning hyper-parameters" begin

        aTolMeanPerError = 0.05
        aTolMaxPerError = 1.2
        aTolScaled = 0.001

        #input
        X = createX()
        #ouptut
        y = createY(X)

        # splitting between train and test samples
        XTrain, XTest, yTrain, yTest = split_train_test(X, y, testRatio = false)

        # scaling
        scaler = StandardScaler()
        fit!(scaler, XTrain)
        XTrainScaled = transform(scaler, XTrain)
        XTestScaled = transform(scaler, XTest)


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

        println("Best hyper-parameters: $(bestParam)")

        # training network using the "best" hidden layer
        #-----------------------------------------------
        clfr = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=bestParam, random_state=1)
        fit!(clfr, XTrainScaled, yTrain)

        # Train set
        #----------
        yPredicted = predict(clfr, XTrainScaled)

        println("Mean Percentage Error Train Set = $(calculate_mean_per_error(yTrain, yPredicted))")
        println("Maximum Ans Percentage Error Train Set = $(calculate_maximum_abs_per_error(yTrain, yPredicted))")
        @test abs(calculate_mean_per_error(yTrain, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_abs_per_error(yTrain, yPredicted)) < aTolMaxPerError


        # Test set
        #---------
        yPredicted = predict(clfr, XTestScaled)

        println("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
        println("Maximum Abs Percentage Error Test Set = $(calculate_maximum_abs_per_error(yTest, yPredicted))")
        @test abs(calculate_mean_per_error(yTest, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_abs_per_error(yTest, yPredicted)) < aTolMaxPerError

        yPredictedScaled = predict_one_obs_scaled(clfr, XTrainScaled[1,:])
        yPredictedUnscaled = predict_one_obs_unscaled(clfr, XTrain[1,:], scaler)

        #scaled versus unscaled observation
        @test yPredictedScaled ≈ yPredictedUnscaled atol = aTolScaled

    end


    #cross-validation to choose the hidden-layer done "manually"
    #-----------------------------------------------------------
    @testset "Testing package on 2d->2d function" begin

        aTolMeanPerError = 0.05
        aTolMaxPerError = 1.2
        aTolScaled = 0.001

        upperBoundX = [1.0; 1.0]
        lowerBoundX = [-1.0; -1.0]


        opts = SModelsOptions(sModelType = :MLPRegressor, classifierType = :MLPClassifier, desiredMinObs = 40)

        surrogatePb = SModelsProblem(
                          lowerBound = lowerBoundX, #lower bound for the parameter space
                          upperBound = upperBoundX,                   #upper bound for the parameter space
                          dimX = 2,                #dimension of the input parameter
                          dimY = 2,                 #dimension of the output vector
                          options = opts)


        set_model_function!(surrogatePb, trueFunction)


        # training the surrogate model
        surrogatem, classifier = train_sModel(surrogatePb, verbose = true, robust = false)

        #=
        #input
        X = createX()
        #ouptut
        y = createY(X)

        # non-robust
        XScaled = transform(surrogatePb.scaler, X)
        yPredicted = predict(surrogatem, XScaled)

        # robust to model failures
        yPredictedRobust = predict_scaled_robust(surrogatem, classifier, XScaled,
                            nonConvergenceFlag = surrogatePb.options.nonConvergenceFlag,
                            convergenceFlag = surrogatePb.options.convergenceFlag,
                            penaltyValue = surrogatePb.options.penaltyValue)

        println("Regressor:")
        println("Mean Percentage Different Set = $(calculate_mean_per_error(y, yPredicted))")
        println("Maximum Abs Percentage Error Different Set = $(calculate_maximum_abs_per_error(y, yPredicted))")

        # Here the model cannot fail, so we should have the same results
        #---------------------------------------------------------------
        @test yPredicted == yPredictedRobust
        =#
    end

    #=
    @testset "Testing package on 2d->2d function that may fail" begin

        aTolMeanPerError = 0.05
        aTolMaxPerError = 1.2
        aTolScaled = 0.001

        upperBoundX = [1.0; 1.0]
        lowerBoundX = [-1.0; -1.0]

        # Need more observations to learn where the model fails
        # When the model fails, the output is a large positive value
        # This creates a large discontinuity
        opts = SModelsOptions(sModelType = :MLPRegressor, classifierType = :MLPClassifier, nbBatches = 5, batchSizeWorker = 100, desiredMinObs = 500)

        surrogatePb = SModelsProblem(    #function f:x -> y that we are trying to approximate
                          lowerBound = lowerBoundX, #lower bound for the parameter space
                          upperBound = upperBoundX,                   #upper bound for the parameter space
                          dimX = 2,                #dimension of the input parameter
                          dimY = 2,                 #dimension of the output vector
                          options = opts)

        set_model_function!(surrogatePb, trueFunctionRisk)

        # training the surrogate model, non-robust method
        surrogatem, classifier = train_sModel(surrogatePb, verbose = true, robust = false)

        # training the surrogate model, robust method
        surrogatemRobust, classifierRobust = train_sModel(surrogatePb, verbose = true, robust = true)

        # Test on a totally new sample:
        #-------------------------------
        #input
        X = createX()
        #ouptut
        y = createYRisk(X, penaltyValue = surrogatePb.options.penaltyValue)

        # non-robust: use normal scaler and ScikitLearn predict function
        #----------------------------------------------------------------
        XScaled = transform(surrogatePb.scaler, X)
        yPredicted = predict(surrogatem, XScaled)

        mean_per_error_non_robust = calculate_mean_per_error(y, yPredicted)
        max_per_error_non_robust = calculate_maximum_abs_per_error(y, yPredicted)

        # robust: use a different scaler
        #-------------------------------
        XScaledRobust = transform(surrogatePb.scalerRobust, X)
        # predict, but first check for convergence using a classifier:
        #-------------------------------------------------------------
        yPredictedRobust = predict_scaled_robust(surrogatemRobust, classifierRobust, XScaledRobust,
                                                nonConvergenceFlag = surrogatePb.options.nonConvergenceFlag,
                                                convergenceFlag = surrogatePb.options.convergenceFlag,
                                                penaltyValue = surrogatePb.options.penaltyValue)

        mean_per_error_robust = calculate_mean_per_error(y, yPredictedRobust)
        max_per_error_robust = calculate_maximum_abs_per_error(y, yPredictedRobust)

        println("Regressor non-robust:")
        println("Mean Percentage Different Set = $(mean_per_error_non_robust)")
        println("Maximum Abs Percentage Error Different Set = $(max_per_error_non_robust)")
        println("----------------------")
        println("Regressor robust:")
        println("Mean Percentage Different Set = $(mean_per_error_robust)")
        println("Maximum Abs Percentage Error Different Set = $(max_per_error_robust)")

    end

    @testset "Testing parallel capabilities" begin

        addprocs(3)

        listMeanPerErr = []
        listMaxPerErr = []

        for i = 1:3

            # with batchSizeWorker = 10
            #--------------------------
            @everywhere using SModels

            @everywhere upperBoundX = [1.0; 1.0]
            @everywhere lowerBoundX = [-1.0; -1.0]

            if i == 1
                @everywhere b = 10
            elseif i==2
                @everywhere b = 100
            else
                @everywhere b = 1000
            end
            @everywhere opts = SModelsOptions(sModelType = :MLPRegressor, batchSizeWorker = b, desiredMinObs = 20, nbBatches = 1)

            @everywhere surrogatePb = SModelsProblem(    #function f:x -> y that we are trying to approximate
                              lowerBound = lowerBoundX,  #lower bound for the parameter space
                              upperBound = upperBoundX,  #upper bound for the parameter space
                              dimX = 2,                  #dimension of the input parameter
                              dimY = 2,                  #dimension of the output vector
                              options = opts)

            # 2d->2d function we aim at fitting
            @everywhere function trueFunction(x::Float64, y::Float64)
                [x^2 + y^2, exp(x)*cos(y)]
            end

            @everywhere function trueFunction(z::Array{Float64,1})
                trueFunction(z[1], z[2])
            end

            set_model_function!(surrogatePb, trueFunction)

            # training the surrogate model
            surrogatem, classifier = train_sModel(surrogatePb, verbose = true, saveToDisk = true)

            #input
            X = createX()
            #ouptut
            y = createY(X)

            XScaled = transform(surrogatePb.scaler, X)

            yPredicted = predict(surrogatem, XScaled)

            append!(listMeanPerErr,calculate_mean_per_error(y, yPredicted))
            append!(listMaxPerErr, calculate_maximum_abs_per_error(y, yPredicted))

            println("Regressor:")
            println("Mean Percentage Different Set = $(calculate_mean_per_error(y, yPredicted))")
            println("Maximum Abs Percentage Error Different Set = $(calculate_maximum_abs_per_error(y, yPredicted))")

        end


        # More points should be converted to more precise surrogate models
        @test abs(listMeanPerErr[1]) > abs(listMeanPerErr[3])
        @test abs(listMaxPerErr[1]) > abs(listMaxPerErr[3])

    end
    =#

end

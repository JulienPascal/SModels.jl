using SModels
using ScikitLearn
using ScikitLearn.GridSearch: GridSearchCV
using JLD, PyCallJLD
@sk_import neural_network: MLPClassifier
@sk_import neural_network: MLPRegressor
@sk_import preprocessing: StandardScaler 

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

srand(1234)

# write your own tests here
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

    @testset "Testing fitting a 2d->2d function" begin

        aTolMeanPerError = 0.05
        aTolMaxPerError = 1.0
        aTolScaled = 0.001

        #input
        X = createX()
        #ouptut
        y = createY(X)

        # splitting between train and test samples
        XTrain, XTest, yTest, yTrain = split_train_test(X, y, testRatio = false)

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
        println("Maximum Percentage Error Train Set = $(calculate_maximum_per_error(yTrain, yPredicted))")
        @test abs(calculate_mean_per_error(yTrain, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_per_error(yTrain, yPredicted)) < aTolMaxPerError


        # Test set
        #---------
        yPredicted = predict(clfr, XTestScaled)

        println("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
        println("Maximum Percentage Error Test Set = $(calculate_maximum_per_error(yTest, yPredicted))")
        @test abs(calculate_mean_per_error(yTest, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_per_error(yTest, yPredicted)) < aTolMaxPerError

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
        XTrain, XTest, yTest, yTrain = split_train_test(X, y, testRatio = false)

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
        println("Maximum Percentage Error Train Set = $(calculate_maximum_per_error(yTrain, yPredicted))")
        @test abs(calculate_mean_per_error(yTrain, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_per_error(yTrain, yPredicted)) < aTolMaxPerError


        # Test set
        #---------
        yPredicted = predict(clfr, XTestScaled)

        println("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
        println("Maximum Percentage Error Test Set = $(calculate_maximum_per_error(yTest, yPredicted))")
        @test abs(calculate_mean_per_error(yTest, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_per_error(yTest, yPredicted)) < aTolMaxPerError

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
        XTrain, XTest, yTest, yTrain = split_train_test(X, y, testRatio = false)

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
            maxPerErr = calculate_maximum_per_error(yTest, yPredicted)

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
        println("Maximum Percentage Error Train Set = $(calculate_maximum_per_error(yTrain, yPredicted))")
        @test abs(calculate_mean_per_error(yTrain, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_per_error(yTrain, yPredicted)) < aTolMaxPerError


        # Test set
        #---------
        yPredicted = predict(clfr, XTestScaled)

        println("Mean Percentage Error Test Set = $(calculate_mean_per_error(yTest, yPredicted))")
        println("Maximum Percentage Error Test Set = $(calculate_maximum_per_error(yTest, yPredicted))")
        @test abs(calculate_mean_per_error(yTest, yPredicted)) < aTolMeanPerError
        @test abs(calculate_maximum_per_error(yTest, yPredicted)) < aTolMaxPerError

        yPredictedScaled = predict_one_obs_scaled(clfr, XTrainScaled[1,:])
        yPredictedUnscaled = predict_one_obs_unscaled(clfr, XTrain[1,:], scaler)

        #scaled versus unscaled observation
        @test yPredictedScaled ≈ yPredictedUnscaled atol = aTolScaled 
        
    end

end

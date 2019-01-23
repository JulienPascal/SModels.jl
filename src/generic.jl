"""
  split_train_test(X::Array{Array{Float64,1},1}, y::Array; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)

Function to split X and Y in between a train and a test sample.
"""
function split_train_test(X::Array{Array{Float64,1},1}, y::Array{Array{Float64,1},1}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)

    uni = Bernoulli(ratio)
    draws = rand(uni, size(X,1))

    XTest = X[draws .== 1]
    XTrain = X[draws .== 0]

    yTest = y[draws .== 1]
    yTrain = y[draws .== 0]

    if testRatio == true
        println("ratio XTest/XTrain = $(size(XTest,1) / size(XTrain,1))")
        println("ratio yTest/yTrain = $(size(yTest,1) / size(yTrain,1))")

        if abs(size(XTest,1)/size(XTrain,1) - ratio) > tolTestRatio
            Base.error("Splitting does not respect the split ratio for X.")
        end
        if abs(size(yTest,1)/size(yTrain,1) - ratio) > tolTestRatio
            Base.error("Splitting does not respect the split ratio for y.")
        end

    end

    return XTrain, XTest, yTrain, yTest

end

"""
  split_train_test(X::Array{Array{Float64,1},1}, y::Array{Array{Float64,1},1}, yConvergence::Array{Int64,1}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)

Function to split X, y and yConvergence in between a train and a test sample.
"""
function split_train_test(X::Array{Array{Float64,1},1}, y::Array{Array{Float64,1},1}, yConvergence::Array{Int64,1}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)

    uni = Bernoulli(ratio)
    draws = rand(uni, size(X,1))

    XTest = X[draws .== 1]
    XTrain = X[draws .== 0]

    yTest = y[draws .== 1]
    yTrain = y[draws .== 0]

    yConvergenceTest = yConvergence[draws .== 1]
    yConvergenceTrain = yConvergence[draws .== 0]

    if testRatio == true

        println("ratio XTest/XTrain = $(size(XTest,1) / size(XTrain,1))")
        println("ratio yTest/yTrain = $(size(yTest,1) / size(yTrain,1))")
        println("ratio yConvergenceTTest/yConvergenceTTrain = $(size(yConvergenceTTest,1) / size(yConvergenceTTrain,1))")

        if abs(size(XTest,1)/size(XTrain,1) - ratio) > tolTestRatio
            Base.error("Splitting does not respect the split ratio for X.")
        end
        if abs(size(yTest,1)/size(yTrain,1) - ratio) > tolTestRatio
            Base.error("Splitting does not respect the split ratio for y.")
        end
        if abs(size(yConvergenceTest,1)/size(yConvergenceTrain,1) - ratio) > tolTestRatio
            Base.error("Splitting does not respect the split ratio for yConvergence.")
        end

    end

    return XTrain, XTest, yTrain, yTest, yConvergenceTrain, yConvergenceTest

end


"""
  split_train_test(X::Array{Float64,2}, y::Array{Float64,2}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)

Function to split X and Y in between a train and a test sample.
"""
function split_train_test(X::Array{Float64,2}, y::Array{Float64,2}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)
  return split_train_test(convert_to_array_of_arrays(X), convert_to_array_of_arrays(y), ratio = ratio, testRatio = testRatio, tolTestRatio = tolTestRatio)
end

"""
  split_train_test(X::Array{Float64,2}, y::Array{Float64,2}, yConvergence::Array{Int64,1}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)

Function to split X, y and yConvergence in between a train and a test sample.
"""
function split_train_test(X::Array{Float64,2}, y::Array{Float64,2}, yConvergence::Array{Int64,1}; ratio::Float64 = 0.25, testRatio::Bool=false, tolTestRatio::Float64 = 0.10)
  return return split_train_test(convert_to_array_of_arrays(X), convert_to_array_of_arrays(y), yConvergence, ratio = ratio, testRatio = testRatio, tolTestRatio = tolTestRatio)
end

"""
  convert_to_array_of_arrays(X::Array{Float64,2})

Function to convert a matrix of type X::Array{Float64,2} to an array of arrays
Array{Array{Float64,1},1}. The latter is the format expected by Scikit-Learn.jl
"""
function convert_to_array_of_arrays(X::Array{Float64,2})

  # initialization
  Y = Array{Array{Float64,1},1}(size(X,1))

  for i=1:size(X,1)
    Y[i] = X[i,:]
  end

  return Y

end

"""
  convert_to_array(X::Array{Array{Float64,1},1})

Function to convert an array of arrays Array{Array{Float64,1},1}
to a matrix of type X::Array{Float64,2} to.
"""
function convert_to_array(X::Array{Array{Float64,1},1})

  # initialization
  Y = zeros(size(X,1), size(X[1],1))

  for j=1:size(X[1],1)
  for i=1:size(X,1)
    Y[i,j] = X[i][j]
  end
  end

  return Y

end

"""
  convert_to_vector(X::Array{Array{Float64,1},1})

Function to convert an array of arrays Array{Array{Float64,1},1}
to a matrix of type X::Array{Float64,1} to.
"""
function convert_to_vector(X::Array{Array{Float64,1},1})

  #safety checks
  if size(X[1],1) > 1
    info("size(X,1) = $(size(X,1))")
    info("X[1] = $(X[1])")
    Base.error("size(X[1],1) = $(size(X[1],1)). Use convert_to_array.")
  end
  # initialization
  Y = zeros(size(X,1))

  for i=1:size(X,1)
    Y[i] = X[i][1]
  end

  return Y

end

"""
  convert_vector_to_array(X::Array{Float64,1})

Scikit-Learn expect the input of the predict function to be of type Array{Float64,2}.
This function converts an Array{Float64,1} to a one-observation Array{Float64,2}.
"""
function convert_vector_to_array(X::Array{Float64,1})

  Y = Array{Float64}(1, length(X))

  for i=1:length(X)
    Y[1,i] = X[i]
  end

  return Y

end

"""
  convert_array_to_vector(X::Array{Float64,2})

Function to convert a one-observation Array{Float64,2} to an Array{Float64,1}
"""
function convert_vector_to_array(X::Array{Float64,2})

  Y = Array{Float64,1}(size(X,2))

  for i=1:size(X,2)
    Y[i] = X[1,i]
  end

  return Y

end

"""
predict_scaled_robust(clfr::PyObject, clf::PyObject, XScaled::Array;
                              nonConvergenceFlag::Int64 = 0,
                              convergenceFlag::Int64 = 1,
                              penaltyValue::Float64 = 999.0)

Function to predict several observations. Robust in the sense that the classifier
is first use to determine whether or not the model converges.
* If the model converges: value is given by the regression model, trained on data
  for which convergence was reached
* If the model does not converge: value = penalty value
"""
function predict_scaled_robust(clfr::PyObject, clf::PyObject, XScaled::Array;
                              nonConvergenceFlag::Int64 = 0,
                              convergenceFlag::Int64 = 1,
                              penaltyValue::Float64 = 999.0)

  # Classifier:
  #------------
  convergenceChek = round(Int64, predict(clf, XScaled))

  # Regression model:
  #------------------
  predictedValues = predict(clfr, XScaled)

  # When the model is not expected to converge, set value = penalty value
  #----------------------------------------------------------------------
  for i=1:size(predictedValues,1)
    if convergenceChek[i] == nonConvergenceFlag
      predictedValues[i, :] = penaltyValue
    end
  end

  return predictedValues

end

"""
  predict_one_obs_scaled(clfr::PyObject, XScaled::Array{Float64,1})

Function to predict one observation. Assume that X has been scaled accordingly.
"""
function predict_one_obs_scaled(clfr::PyObject, XScaled::Array{Float64,1})

  convert_vector_to_array(predict(clfr, convert_vector_to_array(XScaled)))

end


"""
  predict_one_obs_unscaled(clfr::PyObject, XScaled::Array{Float64,1})

Function to predict one observation. Scale the observation first.
"""
function predict_one_obs_unscaled(clfr::PyObject, X::Array{Float64,1}, scaler::PyObject)

  XScaled = transform(scaler, convert_vector_to_array(X))
  convert_vector_to_array(predict(clfr, convert_vector_to_array(XScaled)))

end

"""
  predict_one_obs_unscaled(clfr::PyObject, X::Array{Array{Float64,1}}, scaler::PyObject)

Function to predict one observation. Scale the observation first.
"""
function predict_one_obs_unscaled(clfr::PyObject, X::Array{Array{Float64,1}}, scaler::PyObject)

  XScaled = transform(scaler, X)
  convert_vector_to_array(predict(clfr, XScaled))

end


"""
  set_model_function!(sModelsProblem::SModelsProblem, f::Function)

Function to set the field sModelsProblem.modelFunction
"""
function set_model_function!(sModelsProblem::SModelsProblem, f::Function)

  sModelsProblem.modelFunction = f

end

"""
  set_bounds!(sModelsProblem::SModelsProblem, lowerBound::Array{Float64,1}, upperBound::Array{Float64,1})
"""
function set_bounds!(sModelsProblem::SModelsProblem, lowerBound::Array{Float64,1}, upperBound::Array{Float64,1})

  sModelsProblem.lowerBound = lowerBound
  sModelsProblem.upperBound = upperBound

end


"""
  calculate_mean_per_error(yTrue::Array{Float64,1}, yPredicted::Array{Float64,1})

Function to calculate the mean percentage error
"""
function calculate_mean_per_error(yTrue::Array{Float64,1}, yPredicted::Array{Float64,1})

  perError = zeros(length(yTrue))
  meanPredicted = mean(yPredicted)

  for i=1:length(yTrue)
    if yPredicted[i] != 0
      perError[i] = (yPredicted[i] - yTrue[i])/yTrue[i]
    else
      perError[i] = (yPredicted[i] - yTrue[i])/meanPredicted
    end
  end

  return mean(perError)

end

"""
  calculate_max_per_error(yTrue::Array{Float64,1}, yPredicted::Array{Float64,1})

Function to calculate the maximum percentage error
"""
function calculate_max_per_error(yTrue::Array{Float64,1}, yPredicted::Array{Float64,1})

  perError = zeros(length(yTrue))
  meanPredicted = mean(yPredicted)

  for i=1:length(yTrue)
    if yPredicted[i] != 0
      perError[i] = (yPredicted[i] - yTrue[i])/yTrue[i]
    else
      perError[i] = (yPredicted[i] - yTrue[i])/meanPredicted
    end
  end

  return maximum(perError)

end


"""
  calculate_mean_per_error(yTrue::Array{Float64,2}, yPredicted::Array{Float64,2})

Function to calculate the mean percentage error
"""
function calculate_mean_per_error(yTrue::Array{Float64,2}, yPredicted::Array{Float64,2})

  perError = zeros(size(yTrue, 1),size(yTrue, 2))
  meanPredicted = mean(yPredicted)

  for j=1:size(yTrue, 2)
    for i=1:size(yTrue, 1)
      if yPredicted[i,j] != 0
        perError[i,j] = (yPredicted[i,j] - yTrue[i,j])/yTrue[i,j]
      else
        perError[i,j] = (yPredicted[i,j] - yTrue[i,j])/meanPredicted[j]
      end

    end
  end

  return mean(mean(perError))

end

"""
  calculate_mean_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,2})

Function to calculate the mean percentage error
"""
function calculate_mean_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,2})

  return calculate_mean_per_error(convert_to_array(yTrue), yPredicted)

end

"""
  calculate_maximum_abs_per_error(yTrue::Array{Float64,1}, yPredicted::Array{Float64,1})

Function to calculate the maximum of the absolute value fo the percentage difference
"""
function calculate_maximum_abs_per_error(yTrue::Array{Float64,1}, yPredicted::Array{Float64,1})

  perError = zeros(length(yTrue))
  meanPredicted = mean(yPredicted)

    for i=1:size(yTrue, 1)
      if yPredicted[i] != 0
        perError[i] = abs((yPredicted[i] - yTrue[i])/yTrue[i])
      else
        perError[i] = abs((yPredicted[i] - yTrue[i])/meanPredicted)
      end
    end

  return maximum(perError)

end

"""
  calculate_maximum_abs_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,1})

Function to calculate the maximum of the absolute value fo the percentage difference
"""
function calculate_maximum_abs_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,1})

  return calculate_maximum_abs_per_error(convert_to_vector(yTrue), yPredicted::Array{Float64,1})

end

"""
  calculate_maximum_abs_per_error(yTrue::Array{Float64,2}, yPredicted::Array{Float64,2})

Function to calculate the maximum of the absolute value fo the percentage difference
"""
function calculate_maximum_abs_per_error(yTrue::Array{Float64,2}, yPredicted::Array{Float64,2})

  perError = zeros(size(yTrue, 1),size(yTrue, 2))
  meanPredicted = mean(yPredicted)

  for j=1:size(yTrue, 2)
    for i=1:size(yTrue, 1)
      if yPredicted[i,j] != 0
        perError[i,j] = abs((yPredicted[i,j] - yTrue[i,j])/yTrue[i,j])
      else
        perError[i,j] = abs((yPredicted[i,j] - yTrue[i,j])/meanPredicted[j])
      end

    end
  end

  return maximum(perError)

end

"""
  calculate_maximum_abs_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,2})

Function to calculate the maximum of the absolute value fo the percentage difference
"""
function calculate_maximum_abs_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,2})

  return calculate_maximum_abs_per_error(convert_to_array(yTrue), yPredicted)

end

"""
  calculate_median_abs_per_error(yTrue::Array{Float64,2}, yPredicted::Array{Float64,2})

Function to calculate the median of the absolute value fo the percentage difference
"""
function calculate_median_abs_per_error(yTrue::Array{Float64,2}, yPredicted::Array{Float64,2})

  perError = zeros(size(yTrue, 1),size(yTrue, 2))
  meanPredicted = mean(yPredicted)

  for j=1:size(yTrue, 2)
    for i=1:size(yTrue, 1)
      if yPredicted[i,j] != 0
        perError[i,j] = abs((yPredicted[i,j] - yTrue[i,j])/yTrue[i,j])
      else
        perError[i,j] = abs((yPredicted[i,j] - yTrue[i,j])/meanPredicted[j])
      end

    end
  end

  return median(perError)

end

"""
  calculate_median_abs_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,2})

Function to calculate the median of the absolute value fo the percentage difference
"""
function calculate_median_abs_per_error(yTrue::Array{Array{Float64,1},1}, yPredicted::Array{Float64,2})

  return calculate_median_abs_per_error(convert_to_array(yTrue), yPredicted)

end


"""
  date_now()

Return date and time
"""
function date_now()

  "$(Dates.today())--$(Dates.hour(Dates.now()))h-$(Dates.minute(Dates.now()))m-$(Dates.second(Dates.now()))s"

end

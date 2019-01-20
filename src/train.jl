# * use DistributedArrays
# *

function train_sModel( ;verbose::Bool=false)


  # Initialization
  XDense = Array{Float64,2}()
  YDense = Array{Float64,2}()

  # Generate random draws from the parameter space
  # [TODO]

  for i=1:nbBatches

    if verbose == true
      info("nbBatches = $(nbBatches)")
    end

    # row dimension = moving along observations
    # column dimension = moving along one observation
    # Model's input:
    #---------------
    XDistributed = dzeros((batchSize, dimX), workers(), [nworkers(), 1])
    # Model's output
    #---------------
    YDistributed = dzeros((batchSize, dimY), workers(), [nworkers(), 1])

    # Split the work on the cluster
    #------------------------------
    @sync for (wIndex, wID) in enumerate(workers())
      # Each worker modifies its part of XDistributed and YDistributed
      @async evaluateModel!(localpart(XDistributed), localpart(YDistributed))
    end

    # Concatenate results
    #---------------------
    # First iteration of the loop is a special case
    if XDense == Array{Float64,2}()
      XDense = convert(Array, XDistributed)
    else
      XDense = vcat(XDense, convert(Array, XDistributed))
    end

    if YDense == Array{Float64,2}()
      YDense = convert(Array, YDistributed)
    else
      YDense = vcat(YDense, convert(Array, YDistributed))
    end

    # Split between train and test samples
    #--------------------------------------
    # TODO

    # Train the model, use crossvalidation
    #-------------------------------------
    # TODO
    if verbose == true
      info("Training the model")
    end

    # Test model accuracy
    # If good enough, stop and save the model
    # Else, move the next batch
    #----------------------------------------
    # TODO
    if verbose == true
      info("Percentage Error = ")
    end

  end

end

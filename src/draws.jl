"""
  create_grid_stochastic(a::Array{Float64,1}, b::Array{Float64,1}, nums::Int64)
Function to create a grid. a is a vector of lower
bounds, b a vector of upper bounds and numPoints is the number of points
to be generated. The output is an Array{Float64,2}, where each row is a new point
and each column is a dimension of this points.
"""
function create_grid_stochastic(a::Array{Float64,1}, b::Array{Float64,1}, numPoints::Int64; gridType::Symbol = :uniform, alpha::Float64 = 0.025)

    #Each row is a new point and each column is a dimension of this points.
    #---------------------------------------------------------------------
    pointsFound = zeros(numPoints, length(a))

	  #Safety checks
	  #-------------
	  if numPoints < 1
	    Base.error("The input numPoints should be >= 1. numPoints = $(numPoints).")
	  end

	  if length(a) != length(b)
	    Base.error("length(a) != length(b)")
	  end

    if sum(a .< b) != length(b)
      Base.error("a .< b should hold")
    end


    # Loop until getting the desired number of points
    #-------------------------------------------------
    nbPointsFound = 0
    #Drawing from a normal
    #----------------------
    if gridType == :normal

      # Let's create a variance-covariance matrix such that the point
      # has 95% chances of being within the upper and upper bounds
      #---------------------------------------------------------------
      arrayStd = generate_std(a, b, numPoints, alpha)
      varCov = diagm(arrayStd.^2)

      d = MvNormal((a .+ b)./2, varCov)

      while nbPointsFound < numPoints
        draws = rand(d, numPoints)

        #Check whether point is within upper and lower bound
        #----------------------------------------------------
        for i=1:size(draws,2)
          # Take the sum to check all the inequalities at once
          if sum(draws[:,i] .>= a) == length(a) && sum(draws[:,i] .<= b) == length(a)
            nbPointsFound +=1
            # Check that the output is not "full"
            if nbPointsFound <= numPoints
              pointsFound[nbPointsFound,:] = draws[:,i]
            end

          end
        end

      end

    # Drawing from uniform
    #---------------------
    elseif gridType == :uniform

      # uniform along each dimension
      #-----------------------------
      # Loop over columns
      for j=1:size(pointsFound,2)
        # Loop over rows
        for i=1:size(pointsFound,1)

          # Remember: each row is a new point and each column is a dimension
          # of this points.
          #------------------------------------------------------------------
          d = Uniform(a[j], b[j])

          pointsFound[i,j] = rand(d, 1)[1]
          nbPointsFound +=1

        end
      end

    else
      Base.error("gridType has to be :normal or :uniform")
    end


    return pointsFound

end


"""
  generate_std(a::Array{Float64,1}, b::Array{Float64,1}, numPoints::Int64, alpha::Float64)
Function to generate standard errors such that there is (1-alpha)% chances for
tge point to fall within the upper and lower bound, when sampling from the normal distribution
"""
function generate_std(a::Array{Float64,1}, b::Array{Float64,1}, numPoints::Int64, alpha::Float64)

  # Output:
  #--------
  arrayStd = zeros(length(b))

  #Safety checks
  #-------------
  if numPoints < 1
    Base.error("The input numPoints should be >= 1. numPoints = $(numPoints).")
  end

  if length(a) != length(b)
    Base.error("length(a) != length(b)")
  end

  if alpha <= 0 && alpha > 1
    Base.error("alpha should be such that 0 < alpha < 1")
  end

  if sum(a .< b) != length(b)
    Base.error("a .< b should hold")
  end

  meanValue = (a .+ b)./2


  for i=1:length(b)

      arrayStd[i] = (sqrt(numPoints)*(b[i]-meanValue[i]))/quantile(Normal(), 1 - (alpha/2))

  end

  return arrayStd

end

module TrainRidge
  """
  ESN training functions using the standard ridge regression.

  Several solvers are implemented, including the default ReservoirComputing
  solver, a closed-form solution, and a couple of iterative solvers for
  comparison.
  """

  export train_strategy

  using LinearAlgebra: I, cond
  using MLJLinearModels
  using Optim
  using ReservoirComputing: ESNtrain, nla


  # public interface

  function train_default(esn; beta=0.0)
    """Trains ESN using built-in ESNTrain function."""
    W_out = ESNtrain(esn, beta)
  end

  function train_closedform(esn; beta=0.0)
    """Trains ESN using closed-form ridge regression."""
    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth

    W_out = Y*X'*inv(X*X'+beta*I)     # analytical ridge regression
  end

  function train_mlj(esn; beta=0.0)
    """Trains ESN using ridge regression with MLJ Linear Solver."""
    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth

    W_out = zeros(Float64,size(Y,1),size(X,1))

    solver = MLJLinearModels.Analytical(iterative=true)
    loss = MLJLinearModels.RidgeRegression(lambda=beta, fit_intercept=false)

    for i=1:size(Y,1)
      W_out[i,:] = MLJLinearModels.fit(loss, X', Y[i,:], solver=MLJLinearModels.Analytical())
    end
    W_out
  end

  function train_optim(esn; beta=0.0, opt=BFGS(), maxiters=1000)
    """Trains ESN using ridge regression with Optim.jl nonlinear optimizer."""

    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth
    W_out = zeros(size(Y,1),size(X,1))

    # loss function and analytical derivative (much faster than autodiff!)
    ridge(W) = 1/2*sum((W*X-Y).^2) + 1/2*beta*sum(W.^2)
    ridge_g!(∇,W) = (∇[:,:] = (W*X-Y)*X' + beta*W)
    ridge_h!(∇²,_) = (∇²[:,:] = X*X' + beta*I)

    results = optimize(ridge, ridge_g!, ridge_h!, W_out, opt,
                       Optim.Options(iterations=maxiters),
                       autodiff=:forward)

    W_out = Optim.minimizer(results)
  end

end # module TrainRidge

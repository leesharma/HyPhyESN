module TrainHybrid

  export train

  using ReservoirComputing: ESNtrain, nla, leaky_fixed_rnn
  using LinearAlgebra: I, cond
  using Optim
  using MLJLinearModels

  include("lorenz.jl")
  using .LorenzData: reduced_lorenz_step


  # public interface

  function train_strategy(strategy=:default)
    """
      train_strategy(:optim)

    Returns an ESN training function. Wrapper for selecting training function
    by key, just to keep things neat.

    Current options:
      * :default - uses ReservoirComputing.jl's default solver
      * :closedform - uses a closed-form ridge regression with regularization
      * :mlj - uses MLJLinearModels.jl to solve a ridge regression iteratively
      * :optim - uses Optim.jl (BGFS by default) to solve a ridge regression
    """
    if strategy==:default
      return train_default
    elseif strategy==:closedform
      return train_closedform
    elseif strategy==:mlj
      return train_mlj
    elseif strategy==:optim
      return train_optim
    else
      error("Unknown training strategy $(strategy)")
    end
  end


  ### internal implementation ###

  # function train_default(esn; beta=0.0)
  #   """Trains ESN using built-in ESNTrain function."""
  #   W_out = ESNtrain(esn, beta)
  # end

  function train_closedform_hybrid(esn; beta=0.0)
    """Trains ESN using closed-form ridge regression."""
    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth

    Y_rom = LorenzData.reduced_lorenz_step(esn.train_data)
    X_hybrid = [X; Y_rom]

    W_out = Y*X_hybrid'*inv(X_hybrid*X_hybrid' + beta*I)   # analytical ridge regression
  end

  function train_mlj_hybrid(esn; beta=0.0)
    """Trains ESN using ridge regression with MLJ Linear Solver."""
    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth

    Y_rom = LorenzData.reduced_lorenz_step(esn.train_data)
    X_hybrid = [X; Y_rom]

    W_out = zeros(Float64,size(Y,1),size(X_hybrid,1))

    solver = MLJLinearModels.Analytical(iterative=true)
    loss = MLJLinearModels.RidgeRegression(lambda=beta, fit_intercept=false)

    for i=1:size(Y,1)
      W_out[i,:] = MLJLinearModels.fit(loss, X_hybrid', Y[i,:], solver=MLJLinearModels.Analytical())
    end
    W_out
  end

  function train_optim_hybrid(esn; beta=0.0, opt=BFGS(), maxiters=1000)
    """Trains ESN using ridge regression with Optim.jl nonlinear optimizer."""

    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth

    Y_rom = LorenzData.reduced_lorenz_step(esn.train_data)
    X_hybrid = [X; Y_rom]

    W_out = zeros(size(Y,1),size(X_hybrid,1))

    # loss function and analytical derivative (much faster than autodiff!)
    ridge(W) = 1/2*sum((W*X_hybrid-Y).^2) + 1/2*beta*sum(W.^2)
    ridge_g!(∇,W) = (∇[:,:] = (W*X_hybrid-Y)*X_hybrid' + beta*W)
    ridge_h!(∇²,_) = (∇²[:,:] = X_hybrid*X_hybrid' + beta*I)

    results = optimize(ridge, ridge_g!, ridge_h!, W_out, opt,
                       Optim.Options(iterations=maxiters),
                       autodiff=:forward)

    W_out = Optim.minimizer(results)
  end

  function HyESNpredict(esn, predict_len, W_out; ϵᵦ=0.05)
      output = zeros(Float64, esn.in_size, predict_len)
      x = esn.states[:, end]

      if esn.extended_states == false
          for i=2:predict_len
              x_new = nla(esn.nla_type, x)

              hybrid_term = LorenzData.reduced_lorenz_step(output[:,i-1],ϵᵦ)
              x_new = [x_new; hybrid_term]


              out = (W_out*x_new)
              output[:, i] = out
              x = leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x, out)
          end
      elseif esn.extended_states == true
          for i=2:predict_len
              x_new = nla(esn.nla_type, x)

              hybrid_term = LorenzData.reduced_lorenz_step(output[:,i-1])
              x_new = hcat(x_new, hybrid_term)

              out = (W_out*x_new)
              output[:, i] = out
              x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, esn.W, esn.W_in, x[1:esn.res_size], out), out)
          end
      end
      return output
  end


end # module TrainHybrid

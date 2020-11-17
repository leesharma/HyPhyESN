module BaseESN
  """Demo of a custom train function using Flux.jl"""

  export run_trial

  using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2, nla
  using LinearAlgebra: I
  using Optim

  # default model parameters
  default_params = (
    approx_res_size = 300,   # size of the reservoir
    radius = 1.2,            # desired spectral radius
    activation = tanh,       # neuron activation function
    degree = 6,              # degree of connectivity of the reservoir
    sigma = 0.1,             # input weight scaling
    beta = 0.0001,           # ridge
    alpha = 1.0,             # leaky coefficient
    nla_type = NLAT2(),      # non linear algorithm for the states
    extended_states = false, # if true extends the states with the input
  )

  # public interface

  function esn_init(train_data; opts=default_params)
    """Create echo state network"""
    ESN(
      opts.approx_res_size,
      train_data,
      opts.degree,
      opts.radius,
      activation = opts.activation,
      sigma = opts.sigma,
      alpha = opts.alpha,
      nla_type = opts.nla_type,
      extended_states = opts.extended_states
    )
  end

  function train(esn; beta=0.0, opt=BFGS(), maxiters=1000)
    """Returns a trained readout layer using ridge regression."""

    X = nla(esn.nla_type, esn.states) # reservoir output
    Y = esn.train_data                # groundtruth
    W_out = zeros(size(Y,1),size(X,1))

    # loss function and derivatives (much faster than autodiff)
    ridge(W) = 1/2*sum((W*X-Y).^2) + 1/2*beta*sum(W.^2)
    ridge_g!(∇,W) = (∇[:,:] = (W*X-Y)*X' + beta*W)
    ridge_h!(∇²,_) = (∇²[:,:] = X'*X * beta*I)

    results = optimize(ridge, ridge_g!, ridge_h!, W_out, opt,
                       Optim.Options(iterations=maxiters),
                       autodiff=:forward)

    W_out = Optim.minimizer(results)
  end

  function predict(esn, predict_len, W_out)
    """Predict output sequence of predict_len using the given ESN and W_out."""
    output = ESNpredict(esn, predict_len, W_out)
  end

  function run_trial(train_data, test_data; opts=default_params)
    """Creates, trains, and evaluates an ESN with the given options."""
    test_len = size(test_data, 2)
    esn = esn_init(train_data, opts=opts)
    W_out = train(esn, beta=opts.beta)

    predict(esn, test_len, W_out)
  end
end

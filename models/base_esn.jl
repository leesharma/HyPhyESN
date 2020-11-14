module BaseESN
  export run_trial

  using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2

  # default model parameters
  default_params = (
    approx_res_size = 300,   # size of the reservoir
    radius = 1.2,            # desired spectral radius
    activation = tanh,       # neuron activation function
    degree = 6,              # degree of connectivity of the reservoir
    sigma = 0.1,             # input weight scaling
    beta = 0.0,              # ridge
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

  function train(esn; beta=0.0)
    """Returns a trained readout layer using ridge regression."""
    W_out = ESNtrain(esn, beta)
  end

  function predict(esn, predict_len, W_out)
    """Predict output sequence of predict_len using the given ESN and W_out."""
    output = ESNpredict(esn, predict_len, W_out)
  end

  function run_trial(train, test; opts=default_params)
    """Creates, trains, and evaluates an ESN with the given options."""
    test_len = size(test, 2)
    esn = esn_init(train, opts=opts)
    W_out = ESNtrain(esn, opts.beta)

    predict(esn, test_len, W_out)
  end
end

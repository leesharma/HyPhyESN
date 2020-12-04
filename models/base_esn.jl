module BaseESN
  export run_trial

  using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2

  include("../models/train_ridge.jl")
  using .TrainRidge: train_default, train_closedform, train_mlj, train_optim
  include("../models/train_hybrid.jl")
  using .TrainHybrid: train_closedform_hybrid, train_mlj_hybrid, train_optim_hybrid

  training_strategies = Dict(
    # base
    :default            => train_default,
    :closedform         => train_closedform,
    :mlj                => train_mlj,
    :optim              => train_optim,
    # hybrid
    :closedform_hybrid  => train_closedform_hybrid,
    :mlj_hybrid         => train_mlj_hybrid,
    :optim_hybrid       => train_optim_hybrid,
  )

  # default model parameters
  default_params = (
    approx_res_size = 500,   # size of the reservoir
    radius = 1.0,            # desired spectral radius
    activation = tanh,       # neuron activation function
    degree = 3,              # degree of connectivity of the reservoir
    sigma = 0.1,             # input weight scaling
    beta = 0.0001,           # ridge
    alpha = 1.0,             # leaky coefficient
    nla_type = NLAT2(),      # non linear algorithm for the states
    extended_states = false, # if true extends the states with the input
  )

  # public interface

  function run_trial(train_data, test_len, strategy=:default; opts=default_params)
    """Creates, trains, and evaluates an ESN with the given options."""
    esn = esn_init(train_data, opts=opts)
    train_fn = train_strategy(strategy)
    W_out = train_fn(esn, beta=opts[:beta])

    predict(esn, test_len, W_out)
  end


  # private implementation

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

  function train_strategy(strategy="default")
    """
      train_strategy(:optim)

    Returns an ESN training function. Wrapper for selecting training function
    by key, just to keep things neat.

    Current options:
      * :default - uses ReservoirComputing.jl's default solver
      * :closedform - uses a closed-form ridge regression with regularization
      * :mlj - uses MLJLinearModels.jl to solve a ridge regression iteratively
      * :optim - uses Optim.jl (BGFS by default) to solve a ridge regression
      * :*_hybrid - uses the above approaches with a hybrid ROM model
    """
    strategy=Symbol(strategy)
    return training_strategies[strategy]
  end

  function predict(esn, predict_len, W_out)
    """Predict output sequence of predict_len using the given ESN and W_out."""
    output = ESNpredict(esn, predict_len, W_out)
  end

  function predict_hybrid(esn, predict_len, W_out; ϵᵦ=0.05)

  end
end

module BaseESN
  export run_trial

  using Test
  using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2,
                            HESN, HESNpredict, init_reservoir_givendeg, physics_informed_input

  include("../models/train_ridge.jl")
  using .TrainRidge: train_default, train_closedform, train_mlj, train_optim
  include("../models/train_hybrid.jl")
  using .TrainHybrid: train_hybrid
  # include("../models/train_phyesn.jl")
  # using .TrainPhyESN: train_doan

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
    # HyESN
    in_size = 6,
    γ = 0.5,
    prior_model_size = 3,
  )

  # public interface

  function run_trial(train_data, test_len, strategy=:default; opts=default_params)
    """Creates, trains, and evaluates an ESN with the given options."""
    esn = esn_init(train_data, opts=opts)
    train_fn = train_strategy(strategy)
    W_out = train_fn(esn, beta=opts[:beta])

    predict(esn, test_len, W_out)
  end

  # TODO: consolidate
  function run_hybrid_trial(train_data, test_len, prior_model, u0=[1.,0.,0.], tspan=(0.,1000.), datasize=size(train_data,1); opts=default_params)
    """Creates, trains, and evaluates an ESN with the given options."""
    hesn = hesn_init(train_data, prior_model, u0, tspan, datasize, opts=opts)
    train_fn = train_strategy(:hybrid) # locked to :hybrid
    W_out = train_fn(hesn, beta=opts[:beta])

    # verify readout size
    out_size = size(train_data,1)
    @test size(W_out) == (out_size, hesn.res_size+size(hesn.physics_model_data,1))

    HESNpredict(hesn, test_len, W_out)
  end

  # internal functions

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

  function hesn_init(train_data, prior_model, u0, tspan, datasize; opts=default_params)
    if opts.extended_states
      error("Extended not supported in hybrid mode (yet!)")
    end

    res_size = Integer(floor(opts.approx_res_size/opts.in_size)*opts.in_size)
    W = init_reservoir_givendeg(res_size, opts.radius, opts.degree)
    W_in = physics_informed_input(opts.approx_res_size, opts.in_size, opts.sigma, opts.γ, opts.prior_model_size)

    hesn = HESN(
      W,
      train_data,
      prior_model,
      u0,
      tspan,
      datasize,
      W_in,
      activation = opts.activation,
      alpha = opts.alpha,
      nla_type = opts.nla_type,
      extended_states = opts.extended_states
    )

    # START DEBUG: verify dimensions - some settings on ESN break HESN
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2]-trange[1]
    @test dt==0.02
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt+tspan[2])
    physics_model_data = prior_model(u0, tspan_new, tsteps)

    @test isequal(Integer(floor(opts.approx_res_size/opts.in_size)*opts.in_size), hesn.res_size)
    @test isequal(train_data, hesn.train_data)
    @test isequal(prior_model(u0, tspan_new, tsteps), hesn.physics_model_data)
    @test isequal(vcat(train_data, physics_model_data[:, 1:end-1]), vcat(hesn.train_data, hesn.physics_model_data[:,1:end-1]))
    @test isequal(opts.alpha, hesn.alpha)
    @test isequal(opts.activation, hesn.activation)
    @test isequal(opts.nla_type, hesn.nla_type)
    @test isequal(prior_model, hesn.prior_model)
    @test isequal(datasize, hesn.datasize)
    @test isequal(u0, hesn.u0)
    @test isequal(tspan, hesn.tspan)
    @test isequal(dt, hesn.dt)
    @test size(hesn.W) == (hesn.res_size, hesn.res_size)
    @test size(hesn.W_in) == (hesn.res_size, hesn.in_size)
    @test size(hesn.states) == (hesn.res_size, size(train_data,2))
    # END DEBUG

    hesn
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
    """
    training_strategies = Dict(
      :default            => train_default,
      :closedform         => train_closedform,
      :mlj                => train_mlj,
      :optim              => train_optim,
      # :doan               => train_doan,
      :hybrid             => train_hybrid,
    )[Symbol(strategy)]
  end

  function predict(esn, predict_len, W_out)
    """Predict output sequence of predict_len using the given ESN and W_out."""
    output = ESNpredict(esn, predict_len, W_out)
  end
end

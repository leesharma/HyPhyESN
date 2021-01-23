module ModelSelection
  """Parameter search utilities"""

  include("../eval/metrics.jl")
  using .Metrics: time_horizon
  include("../models/base_esn_optim.jl")
  using .BaseESN: run_trial

  using Printf: @printf
  using ReservoirComputing: NLAT2
  using Statistics: mean, std


  function random_search(train, test, grid_params; n_samples=4, n_trials=10, E_max=0.4, dt=0.02)
    """
      random_search(train, test, params; n_samples=20, n_trials=50)

    Run a random parameter search from the given params, with `n_samples` random
    samples and `n_trials` repeated trials per parameter set.

    Prints out mean and standard deviation for each parameter set.

    Example:

      julia> random_search( train,
                            test,
                            Dict(:approx_res_size=>[200,300],:radius=[1.0,1.2]),
                            n_samples=20,
                            n_trials=100 )
    """

    # Runs a trial and returns the time horizon for the given options
    #   note: we could inject the metric if there are others that make sense
    time_horizon(opts) = Metrics.time_horizon(test, BaseESN.run_trial(train,test,opts=opts), E_max=E_max, dt=dt)
    # Randomly samples from a list, falling back to the given default
    sample(key;default) = rand(get(grid_params,key,[default]))

    for _ in 1:n_samples
      # build options
      opts = (
        approx_res_size = sample(:approx_res_size, default=300),  # size of the reservoir
        radius = sample(:radius, default=1.2),                    # desired spectral radius
        activation = sample(:activation, default=tanh),           # neuron activation function
        degree = sample(:degree, default=6),                      # degree of connectivity of the reservoir
        sigma = sample(:sigma, default=0.1),                      # input weight scaling
        beta = sample(:beta, default=0.0),                        # ridge
        alpha = sample(:alpha, default=1.0),                      # leaky coefficient
        nla_type = sample(:nla_type, default=NLAT2()),            # non linear algorithm for the states
        extended_states = false,                                  # if true extends the states with the input
      )

      # run trials
      time_horizons = [time_horizon(opts) for _ in 1:n_trials]

      # print results
      _print_results(opts, time_horizons, n_trials)
    end
  end

  function grid_search(train, test, grid_params; n_trials=50, E_max=0.4, dt=0.02)
    """
      grid_search(train, test, grid_params; n_samples=20, n_trials=50)

    Run a grid search from the given params, with `n_trials` repeated trials for
    each parameter set.

    Prints out mean and standard deviation for each parameter set.
    """

    # Runs a trial and returns the time horizon for the given options
    #   note: we could inject the metric if there are others that make sense
    time_horizon(opts) = Metrics.time_horizon(test, BaseESN.run_trial(train,test,opts=opts), E_max=E_max, dt=dt)
    # Syntactic sugar for Base.get(collection, key, default)
    options(key;default) = get(grid_params,key,[default])

    for approx_res_size in options(:approx_res_size, default=300)
      for radius in options(:radius, default=1.2)
        for activation in options(:activation, default=tanh)
          for degree in options(:degree, default=6)
            for sigma in options(:sigma, default=0.1)
              for beta in options(:beta, default=0.0)
                for alpha in options(:alpha, default=1.0)
                  for nla_type in options(:nla_type, default=NLAT2())
                    # build options
                    opts = (
                      approx_res_size = approx_res_size,  # size of the reservoir
                      radius = radius,                    # desired spectral radius
                      activation = activation,            # neuron activation function
                      degree = degree,                    # degree of connectivity of the reservoir
                      sigma = sigma,                      # input weight scaling
                      beta = beta,                        # ridge
                      alpha = alpha,                      # leaky coefficient
                      nla_type = nla_type,                # non linear algorithm for the states
                      extended_states = false,            # if true extends the states with the input
                    )

                    # run trials
                    time_horizons = [time_horizon(opts) for _ in 1:n_trials]

                    # print results
                    _print_results(opts, time_horizons, n_trials)
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  function _print_results(opts, results, n_trials; metric_name="time horizon")
    """Prints options and statistics for given results."""

    display(opts)
    @printf(
      "  Mean %s:  %0.2f ± %0.1f Lynapunov times (n=%d)\n",
      metric_name, mean(results), std(results), n_trials
    )
  end

end # ModelSelection

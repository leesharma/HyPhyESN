module ModelSelection
  """Parameter search utilities"""

  include("../eval/metrics.jl")
  using .Metrics: time_horizon
  include("../models/base_esn.jl")
  using .BaseESN: run_trial

  using Printf: @printf
  using ReservoirComputing: NLAT2
  using Statistics: mean, std, quantile


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

    test_len = size(test,2)
    # Runs a trial and returns the time horizon for the given options
    #   note: we could inject the metric if there are others that make sense
    time_horizon(opts) = Metrics.time_horizon(test, BaseESN.run_trial(train,test_len,:closedform,opts=opts), E_max=E_max, dt=dt)
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

    test_len = size(test,2)
    # Runs a trial and returns the time horizon for the given options
    #   note: we could inject the metric if there are others that make sense
    time_horizon(opts) = Metrics.time_horizon(test, BaseESN.run_trial(train,test_len,:closedform,opts=opts), E_max=E_max, dt=dt)
    # Syntactic sugar for Base.get(collection, key, default)
    options(key;default) = get(grid_params,key,[default])

    means = zeros(8,6)
    stds = zeros(8,6)

    for (res_idx,approx_res_size) in enumerate(options(:approx_res_size, default=300))
      for (rad_idx,radius) in enumerate(options(:radius, default=1.2))
        for (activation_idx,activation) in enumerate(options(:activation, default=tanh))
          for (deg_idx,degree) in enumerate(options(:degree, default=6))
            for (sigma_idx,sigma) in enumerate(options(:sigma, default=0.1))
              for (beta_idx,beta) in enumerate(options(:beta, default=0.0001))
                for (alpha_idx,alpha) in enumerate(options(:alpha, default=1.0))
                  for (nla_idx,nla_type) in enumerate(options(:nla_type, default=NLAT2()))
                    for (es_idx,extended_states) in enumerate(options(:extended_states, default=false))
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
                        extended_states = extended_states,  # if true extends the states with the input
                      )

                      # run trials
                      try
                        time_horizons = [time_horizon(opts) for _ in 1:n_trials]
                        means[rad_idx,alpha_idx] = mean(time_horizons)
                        stds[rad_idx,alpha_idx] = std(time_horizons)
                        _print_results(opts, time_horizons, n_trials)
                      catch err
                        display(err)
                        means[rad_idx,alpha_idx] = NaN
                        stds[rad_idx,alpha_idx] = NaN
                        display(opt)
                        display(NaN)
                      end

                      # print results
                      # _print_results(opts, time_horizons, n_trials)
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
    means,stds
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


# dt = 0.02
# train_len = 3000
# test_len = 750    # ~14 Lyapunov times
# E_max = 0.4       # error threshold for time horizon
# train, test = LorenzData.train_test(train_len=train_len, predict_len=test_len, dt=dt)

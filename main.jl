# Initialize the environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DataFrames    # for easy data formatting and statistics
using Statistics: mean, std
using Printf: @printf
using BenchmarkTools: @benchmark
using ReservoirComputing: NLAT2

# local imports
include("./data/lorenz.jl")
using .LorenzData: train_test

include("./models/base_esn.jl")
using .BaseESN: run_trial

include("./eval/metrics.jl")
using .Metrics: plot_predictions, plot_error, time_horizon


function main()

  # Create Dataset

  # Note: Doan et al. (2019) used:
  #   - N_t = 1000 training points
  #   - N_p = 1000 testing points
  #   - dt = 0.01 seconds
  dt = 0.02
  train_len = 3000
  test_len = Int(floor(27/dt))    # ~14 Lyapunov times
  E_max = 0.4                     # error threshold for time horizon
  train, test = LorenzData.train_test(train_len=train_len, predict_len=test_len, dt=dt)

  # Describe the Data

  println("Lorenz System")
  println("-------------")
  println("Data shape:")
  println("  train: $(size(train))")
  println("  test:  $(size(test))")
  println()
  println()

  # Run Trials for Metrics

  ##############################################################################
  # START RUN CUSTOMIZATION
  ##############################################################################
  #
  #
  n_trials = 100
  strategy = "closedform"       # select from [:default, :closedform, :mlj, :optim]
  opts = (
    approx_res_size = 200,   # size of the reservoir
    radius = 0.4,            # desired spectral radius
    activation = tanh,       # neuron activation function
    degree = 3,              # degree of connectivity of the reservoir
    sigma = 0.15,            # input weight scaling
    beta = 1e-6,             # ridge
    alpha = 0.7,             # leaky coefficient
    nla_type = NLAT2(),      # non linear algorithm for the states
    extended_states = true,  # if true extends the states with the input
  )
  #
  #
  ##############################################################################
  # END RUN CUSTOMIZATION
  ##############################################################################

  train, test = LorenzData.train_test(train_len=train_len, predict_len=test_len, dt=dt)

  println("Running $(n_trials) trials...")
  max_time_horizon = 0
  best_predictions = zeros(3,test_len)
  time_horizons = zeros(n_trials)
  for i in 1:n_trials
    predictions = BaseESN.run_trial(train,test_len,opts=opts)
    time_horizon = Metrics.time_horizon(test, predictions, E_max=E_max, dt=dt)
    if time_horizon > max_time_horizon
      max_time_horizon = time_horizon
      best_predictions = predictions
    end
    time_horizons[i] = time_horizon
  end
  @printf(
    "  Average time horizon:  %0.2f ± %0.1f Lynapunov times,  Max time horizon:  %0.2f (n=%d)\n",
    mean(time_horizons), std(time_horizons), max(time_horizons...), n_trials
  )
  println()

  # Benchmark Trial

  println("Benchmarking train/predict time...")
  display(@benchmark BaseESN.run_trial($train, $test_len, $strategy, opts=$opts))
  println()

  # Plot Sample Run

  println("Plotting plots...")
  display(Metrics.plot_predictions(test, best_predictions, dt=dt, outfile="output/best_predictions.pdf"))
  display(Metrics.plot_error(test, best_predictions, E_max=E_max, dt=dt, outfile="output/best_error.pdf"))

  test_len = 3000  # longer resolution time for time-avg graph
  train, test = LorenzData.train_test(train_len=train_len, predict_len=test_len, dt=dt)
  predictions = BaseESN.run_trial(train, test_len, strategy, opts=opts)
  display(Metrics.plot_avg_error(test, predictions, E_max=E_max, dt=dt))
end


main()

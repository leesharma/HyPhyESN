# Initialize the environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DataFrames    # for easy data formatting and statistics
using Statistics: mean, std
using Printf:@printf
using BenchmarkTools:@benchmark

# local imports
include("./data/lorenz.jl")
using .LorenzData: train_test

include("./models/base_esn.jl")
using .BaseESN: run_trial

include("./eval/metrics.jl")
using .Metrics: plot_predictions, plot_error


function main()

  # Create Dataset

  # Note: Doan et al. (2019) used:
  #   - N_t = 1000 training points
  #   - N_p = 1000 testing points
  #   - dt = 0.01 seconds
  dt = 0.02
  train_len = 3000
  test_len = 600
  train, test = LorenzData.train_test(train_len=train_len, predict_len=test_len, dt=dt)

  # Describe the Data

  println("Lorenz System")
  println("-------------")
  println("Data shape:")
  println("  train: $(size(train))")
  println("  test:  $(size(test))")
  println()
  println("Training data:")
  display(first(DataFrame(train', ["x","y","z"]), 8))
  println()
  println("Testing data:")
  display(DataFrame(test', ["x","y","z"]))
  println()
  println()

  # Run Trials for Metrics

  n_trials = 50
  println("Running $(n_trials) trials...")
  E_max = 0.4  # error threshold for time horizon
  time_horizons = [
    Metrics.time_horizon(test, BaseESN.run_trial(train,test), E_max=E_max, dt=dt)
    for _ in 1:n_trials
  ]
  @printf(
    "  Average time horizon:  %0.2f ± %0.1f Lynapunov times (n=%d)\n",
    mean(time_horizons), std(time_horizons), n_trials
  )
  println()

  # Benchmark Trial

  println("Benchmarking train/predict time...")
  display(@benchmark BaseESN.run_trial($train, $test))
  println()

  # Plot Sample Run

  println("Plotting plots...")
  predictions = BaseESN.run_trial(train, test)
  display(Metrics.plot_predictions(test, predictions, dt=dt))
  display(Metrics.plot_error(test, predictions, E_max=E_max, dt=dt))

end


main()

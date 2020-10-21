# Initialize the environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DataFrames    # for easy data formatting and statistics

# local imports
include("./data/lorenz.jl")
using .LorenzData

include("./models/base_esn.jl")
using .BaseESN

include("./eval/metrics.jl")
using .Metrics: plot_predictions, plot_errors


function main()

  # Data Loading

  train, test = LorenzData.train_test(train_len=3000, predict_len=1000)

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

  # Run the Model

  esn = BaseESN.esn(train)
  W_out = BaseESN.train(esn)
  output = BaseESN.predict(esn, 1000, W_out)

  println("Predictions:")
  display(DataFrame(output', ["x","y","z"]))
  println()
  plot_predictions(test, output)
  # plot_errors(X_test, output)

end


main()

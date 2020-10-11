# Initialize the environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DataFrames    # for easy data formatting and statistics

# local imports
include("./data/lorenz.jl")
using .LorenzData

include("./models/base_esn.jl")
using .BaseESN


function main()

  # Data Loading

  train, test = LorenzData.train_test(train_len=3000, predict_len=5)

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
  output = BaseESN.predict(esn, 5, W_out)

  println("Predictions:")
  display(DataFrame(output', ["x","y","z"]))
  println()

end


main()

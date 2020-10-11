# Initialize the environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DataFrames    # for easy data formatting and statistics

# local imports
include("./data/lorenz.jl")
using .LorenzData


function main()
  train, test = LorenzData.train_test(train_len=8, predict_len=5)

  println("Lorenz System")
  println("-------------")
  println("Data shape:")
  println("  train: $(size(train))")
  println("  test:  $(size(test))")
  println()
  println("Training data:")
  display(DataFrame(train', ["x","y","z"]))
  println()
  println("Testing data:")
  display(DataFrame(test', ["x","y","z"]))
  println()
end


main()

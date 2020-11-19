include("../data/barotropic.jl")
using .BarotropicData

include("../models/base_esn.jl")
using .BaseESN

###############################################################################
# Data generation parameters
# T21 grid
num_fourier = 21  # Fourier wave number truncation
nθ = 32  # Latitudinal grid size
nd = 20  # Vertical slices, doesn't actually matter here, it just solves nd = 1

start_time = 0
end_time = 691200
Δt = 1800
###############################################################################
# ESN training parameters. Currently throws away first 1/4 of data, trains with next 1/2, tests on final 1/4.
train_len = floor(Int64, (end_time/Δt)/2)
predict_len = floor(Int64, (end_time/Δt)/4)
shift = floor(Int64, (end_time/Δt)/4)
approx_res_size = 3000  # NOTE this must be larger than size(train_u)[1]
###############################################################################


# Generate the data
train_u, test_u, train_v, test_v = BarotropicData.train_test(train_len, predict_len, shift, num_fourier,
                                                            nθ, nd, start_time, end_time, Δt)
# Train the esn
# TODO: Step 1 -  Convert back to grids, plot, and evaluate.
# TODO: Step 2 - stack u & v in dim=1. Get ESN working, then eval results.
esn = BaseESN.esn(train_u, approx_res_size)
W_out = BaseESN.train(esn)
output = BaseESN.predict(esn, predict_len, W_out)

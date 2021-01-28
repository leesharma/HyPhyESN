using Pkg; Pkg.activate("."); Pkg.instantiate()

include("../data/barotropic.jl")
using .BarotropicData

include("../models/base_esn.jl")
using .BaseESN

include("../eval/primitive_metrics.jl")
using .PrimitivesMetrics

using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2
using JGCM
using JLD
using Statistics

###############################################################################
#-- Training parameters
dataset_filepath = "./data/datasets/barotropic_T21_2D_8day.jld"
save_name = "barotropic_T21_2D_baseESN_5Kres_MOD2_norm_HPCTEST"  # Name of file to save results to

model_params = (
  approx_res_size = 5000,   # size of the reservoir; NOTE: Must be larger than all of input params.
  radius = 1.0,              # desired spectral radius
  activation = tanh,         # neuron activation function
  degree = 3,                # degree of connectivity of the reservoir
  sigma = 0.1,               # input weight scaling
  beta = 0.0001,             # ridge
  alpha = 0.6,               # leaky coefficient
  nla_type = NLAT2(),        # non linear algorithm for the states
  extended_states = false,   # if true extends the states with the input
)

# Load some parameters from the dataset
end_time = load(dataset_filepath)["end_time"]
Δt = load(dataset_filepath)["Δt"]

# ESN training parameters. Currently throws away first 1/4 of data, trains with next 1/2, tests on final 1/4.
train_len = floor(Int64, (end_time/Δt)/2)
predict_len = ceil(Int64, (end_time/Δt)/4)
shift = floor(Int64, (end_time/Δt)/4)
###############################################################################


# Load the data
train_u, test_u, train_v, test_v, mesh = BarotropicData.train_test(dataset_filepath,
                                                                   train_len = train_len,
                                                                   predict_len = predict_len,
                                                                   shift = shift)
nθ = mesh.nθ
nd = mesh.nd

u_mean = mean(train_u)
u_std = std(train_u)
v_mean = mean(train_v)
v_std = std(train_v)

#Standardize the input data
train_u = (train_u.-u_mean)./u_std
train_v = (train_v.-v_mean)./v_std

train_data = cat(train_u, train_v, dims=1)

# Initialize ESN, then train, & predict
esn = @time BaseESN.large_esn_init(train_data, opts=model_params)
println("ESN initialized. ...")
W_out = @time BaseESN.train(esn, beta=model_params.beta)
println("ESN trained. ...")
prediction = @time BaseESN.large_predict(esn, predict_len, W_out)
println("Predictions completed. ...")

# Separate u & v from prediction array, reshape them to grid shape
prediction_u = prediction[1:Int64(size(prediction)[1]/2),:]
prediction_v = prediction[Int64(size(prediction)[1]/2+1):end,:]
pred_u_grid = reshape(prediction_u, (2*nθ, nθ, :))
test_u_grid = reshape(test_u, (2*nθ, nθ, :))
pred_v_grid = reshape(prediction_v, (2*nθ, nθ, :))
test_v_grid = reshape(test_v, (2*nθ, nθ, :))

# Undo standardization to compare results
pred_u_grid = (pred_u_grid.*u_std).+u_mean
pred_v_grid = (pred_v_grid.*v_std).+v_mean

# Modify param list for save compatibility (can't save directly with JLD)
save_model_params = (
  approx_res_size = model_params.approx_res_size,
  radius = model_params.radius,
  activation = String(Symbol(model_params.activation)),
  degree = model_params.degree,
  sigma = model_params.sigma,
  beta = model_params.beta,
  alpha = model_params.alpha,
  nla_type = String(Symbol(model_params.nla_type)),
  extended_states = model_params.extended_states,
)

# Save the results
save("./train/results/$save_name.jld","model_params",save_model_params,"pred_u_grid",pred_u_grid,
     "test_u_grid",test_u_grid,"pred_v_grid",pred_v_grid,"test_v_grid",test_v_grid,
     "W_out",W_out,compress = true)
println("Results saved. ...")

barotropic_evaluate(dataset_filepath, save_name)
println("Model evaluation completed. ...")

# Plot the first timestep prediction & ground truth for quick peek
# Lat_Lon_Pcolormesh(mesh, pred_u_grid,  1, "./train/plots/baseESN_barotropic_2D_pred_u_20Kres_MOD1.png")
# Lat_Lon_Pcolormesh(mesh, test_u_grid, 1, "./train/plots/baseESN_barotropic_2D_test_u.png")
# Lat_Lon_Pcolormesh(mesh, pred_v_grid,  1, "./train/plots/baseESN_barotropic_2D_pred_v_20Kres_MOD1.png")
# Lat_Lon_Pcolormesh(mesh, test_v_grid, 1, "./train/plots/baseESN_barotropic_2D_test_v.png")

println("Completed!")

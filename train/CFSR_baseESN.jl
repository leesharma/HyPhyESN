using Pkg; Pkg.activate("."); Pkg.instantiate()

include("../data/spectral_dynamics.jl")
using .SpectralData

include("../models/base_esn.jl")
using .BaseESN

include("../eval/primitive_metrics.jl")
using .PrimitivesMetrics

using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2, Ridge
using JGCM
using JLD
using Statistics
using MLJLinearModels

###############################################################################
#-- Training parameters
dataset_filepath = "./data/datasets/CFSR_T62_2year_3height.jld"
# dataset_filepath = "E:/HyPhyESN_Datasets/CFSR/T62/CFSR_T62_2year_3height.jld"
save_name = "CFSR_T63_2year_3height_baseESN_ver3"  # Name of file to save results to

model_params = (
  approx_res_size = 110000,   # size of the reservoir; NOTE: Must be larger than all of input params.
  radius = 1.0,              # desired spectral radius
  activation = tanh,         # neuron activation function
  degree = 3,                # degree of connectivity of the reservoir
  sigma = 0.1,               # input weight scaling
  beta = 0.0001,             # ridge
  alpha = 0.6,               # leaky coefficient
  nla_type = NLAT2(),        # non linear algorithm for the states
  extended_states = false,   # if true extends the states with the input
)

# Set the ratio of data for training, and ratio for testing.
train_ratio = 3/4
predict_ratio = 1/4

###############################################################################

# Load the data
temporal_grid_u = load(dataset_filepath)["temporal_grid_u"]
temporal_grid_v = load(dataset_filepath)["temporal_grid_v"]
temporal_grid_P = load(dataset_filepath)["temporal_grid_P"]
temporal_grid_T = load(dataset_filepath)["temporal_grid_T"]

# Grab information from the dataset
total_time = size(temporal_grid_u)[4]
nθ = size(temporal_grid_u)[2]
nλ = size(temporal_grid_u)[1]
nd = size(temporal_grid_u)[3]

# Convert day parameters to seconds, then divide by time step to get array indices
train_len = floor(Int64, train_ratio*total_time)
predict_len = floor(Int64, predict_ratio*total_time)

# flatten the spatial components. Leaves us a 2D array of spatial solutions & time step
data_u = reshape(temporal_grid_u[:,:,:,:], (:,size(temporal_grid_u)[4]))
data_v = reshape(temporal_grid_v[:,:,:,:], (:,size(temporal_grid_v)[4]))
data_P = reshape(temporal_grid_P[:,:,:,:], (:,size(temporal_grid_P)[4]))
data_T = reshape(temporal_grid_T[:,:,:,:], (:,size(temporal_grid_T)[4]))

# Clear Memory
temporal_grid_u = 0
temporal_grid_v = 0
temporal_grid_P = 0
temporal_grid_T = 0

# Define train and test sets
train_u = data_u[:, 1:train_len-1]
train_v = data_v[:, 1:train_len-1]
train_P = data_P[:, 1:train_len-1]
train_T = data_T[:, 1:train_len-1]
test_u = data_u[:, train_len:train_len+predict_len-1]
test_v = data_v[:, train_len:train_len+predict_len-1]
test_P = data_P[:, train_len:train_len+predict_len-1]
test_T = data_T[:, train_len:train_len+predict_len-1]

# Clear Memory
data_u = 0
data_v = 0
data_P = 0
data_T = 0


println("Data loaded. ...")

u_mean = mean(train_u)
u_std = std(train_u)
v_mean = mean(train_v)
v_std = std(train_v)
P_mean = mean(train_P)
P_std = std(train_P)
T_mean = mean(train_T)
T_std = std(train_T)

#Standardize the input data
train_u = (train_u.-u_mean)./u_std
train_v = (train_v.-v_mean)./v_std
train_P = (train_P.-P_mean)./P_std
train_T = (train_T.-T_mean)./T_std

# Combine training data.
# NOTE: Moved P to the end of this flattened array for easier indexing. Only has one vertical dimension.
train_data = cat(train_u, train_v, train_T, train_P, dims=1)

# Initialize ESN, then train, & predict

esn = @time BaseESN.large_esn_init(train_data, opts=model_params)
println("ESN initialized. ...")
W_out = @time BaseESN.train(esn, beta=model_params.beta)

#-- For iterative ridge solver, use the below block and comment out .train() above.
# solver = MLJLinearModels.Analytical(iterative=true)
# ridge = Ridge(model_params.beta, solver)
# W_out = @time BaseESN.ESNtrain(ridge, esn)

println("ESN trained. ...")

# Convert predict_len to timesteps for .predict()
prediction = @time BaseESN.large_predict(esn, predict_len, W_out)
println("Predictions completed. ...")

# Separate u, v, P, T from prediction array, reshape them to grid shape
# NOTE: Moved P to the end of this flattened array for easier indexing.
prediction_u = prediction[1:Int64(nλ*nθ*nd),:]
prediction_v = prediction[Int64(nλ*nθ*nd+1):Int64(2*nλ*nθ*nd),:]
prediction_T = prediction[Int64(2*nλ*nθ*nd+1):Int64(3*nλ*nθ*nd),:]
prediction_P = prediction[Int64(3*nλ*nθ*nd+1):end,:]
pred_u_grid = reshape(prediction_u, (nλ, nθ, nd, :))
test_u_grid = reshape(test_u, (nλ, nθ, nd, :))
pred_v_grid = reshape(prediction_v, (nλ, nθ, nd, :))
test_v_grid = reshape(test_v, (nλ, nθ, nd, :))
pred_P_grid = reshape(prediction_P, (nλ, nθ, 1, :))
test_P_grid = reshape(test_P, (nλ, nθ, 1, :))
pred_T_grid = reshape(prediction_T, (nλ, nθ, nd, :))
test_T_grid = reshape(test_T, (nλ, nθ, nd, :))

# Undo standardization to compare results
pred_u_grid = (pred_u_grid.*u_std).+u_mean
pred_v_grid = (pred_v_grid.*v_std).+v_mean
pred_P_grid = (pred_P_grid.*P_std).+P_mean
pred_T_grid = (pred_T_grid.*T_std).+T_mean

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

#-- Set where to save the model once trained
model_filepath = "./train/results/$save_name.jld"

# Save the results
save(model_filepath,"model_params",save_model_params,"pred_u_grid",pred_u_grid,
     "test_u_grid",test_u_grid,"pred_v_grid",pred_v_grid,"test_v_grid",test_v_grid,
     "pred_P_grid",pred_P_grid,"test_P_grid",test_P_grid,"pred_T_grid",pred_T_grid,
     "test_T_grid",test_T_grid,"W_out",W_out,compress = true)
println("Results saved. ...")

# CFSR_evaluate(dataset_filepath, model_filepath, save_name)
# println("Model evaluation completed. ...")

# Plot the prediction & ground truth for quick peek
# time_step = 1 # Time step to plot
# height = 1 # Set the height layer to plot
# Lat_Lon_Pcolormesh(mesh, pred_u_grid[:,:,:,time_step],  height, "./train/plots/baseESN_spectral_nd3_pred_u_35Kres.png")
# Lat_Lon_Pcolormesh(mesh, test_u_grid[:,:,:,time_step], height, "./train/plots/baseESN_spectral_nd3_test_u.png")
# Lat_Lon_Pcolormesh(mesh, pred_v_grid[:,:,:,time_step],  height, "./train/plots/baseESN_spectral_nd3_pred_v_35Kres.png")
# Lat_Lon_Pcolormesh(mesh, test_v_grid[:,:,:,time_step], height, "./train/plots/baseESN_spectral_nd3_test_v.png")
# Lat_Lon_Pcolormesh(mesh, pred_P_grid[:,:,:,time_step],  height, "./train/plots/baseESN_spectral_nd3_pred_P_35Kres.png")
# Lat_Lon_Pcolormesh(mesh, test_P_grid[:,:,:,time_step], height, "./train/plots/baseESN_spectral_nd3_test_P.png")
# Lat_Lon_Pcolormesh(mesh, pred_T_grid[:,:,:,time_step],  height, "./train/plots/baseESN_spectral_nd3_pred_T_35Kres.png")
# Lat_Lon_Pcolormesh(mesh, test_T_grid[:,:,:,time_step], height, "./train/plots/baseESN_spectral_nd3_test_T.png")
# println("Results plotted. ...")

# Countour plot if desired, not currently setup correctly.
#Sigma_Zonal_Mean_Contourf(op_man, "./data/data_plots/spectral_dynamics_contourf")

println("Run complete!")

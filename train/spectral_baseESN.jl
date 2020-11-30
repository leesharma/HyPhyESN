include("../data/spectral_dynamics.jl")
using .SpectralData

include("../models/base_esn.jl")
using .BaseESN

using ReservoirComputing: ESN, ESNtrain, ESNpredict, NLAT2
using JGCM
using JLD

###############################################################################
#-- Training parameters
dataset_filepath = "./data/datasets/spectral_T21_600day_200spinup.jld"
save_name = "spectral_T21_baseESN.jld"  # Name of file to save results to

model_params = (
  approx_res_size = 10000,   # size of the reservoir; NOTE: Must be larger than all of input params.
  radius = 1.0,              # desired spectral radius
  activation = tanh,         # neuron activation function
  degree = 3,                # degree of connectivity of the reservoir
  sigma = 0.1,               # input weight scaling
  beta = 0.0001,             # ridge
  alpha = 1.0,               # leaky coefficient
  nla_type = NLAT2(),        # non linear algorithm for the states
  extended_states = false,   # if true extends the states with the input
)

# Get end day & spinup day parameters from the dataset.
op_man = load(dataset_filepath)["op_man"]
end_day = op_man.end_time/86400
spinup_day = op_man.spinup_day
mesh = load(dataset_filepath)["mesh"]

# Spinup_day is thrown away. Of remaining, 3/4 is training, 1/4 is predict.
# Can change this to whatever you want.
train_len = floor(Int64, (end_day-spinup_day)*(3/4))
predict_len = ceil(Int64, (end_day-spinup_day)*(1/4))
###############################################################################


# Load the data
op_man, train_u, train_v, train_P, train_T, test_u, test_v, test_P, test_T = SpectralData.train_test(dataset_filepath, train_len, predict_len)
println("Data loaded. ...")
# Combine u & v training data
#train_data = cat(train_u, train_v, train_P, train_T, dims=1)
train_data = cat(train_u, train_v, train_P, train_T, dims=1)
nθ = mesh.nθ

# Initialize ESN, then train, & predict
esn = BaseESN.esn_init(train_data, opts=model_params)
println("ESN initialized. ...")
W_out = BaseESN.train(esn, beta=model_params.beta)
println("ESN trained. ...")
prediction = BaseESN.predict(esn, predict_len, W_out)
println("Predictions completed. ...")

# Separate u, v, P, T from prediction array, reshape them to grid shape
prediction_u = prediction[1:Int64(size(prediction)[1]/4),:]
prediction_v = prediction[Int64(size(prediction)[1]/4+1):Int64(size(prediction)[1]/2),:]
prediction_P = prediction[Int64(size(prediction)[1]/2+1):Int64(size(prediction)[1]*3/4),:]
prediction_T = prediction[Int64(size(prediction)[1]*3/4+1):end,:]
pred_u_grid = reshape(prediction_u, (2*nθ, nθ, :))
test_u_grid = reshape(test_u, (2*nθ, nθ, :))
pred_v_grid = reshape(prediction_v, (2*nθ, nθ, :))
test_v_grid = reshape(test_v, (2*nθ, nθ, :))
pred_P_grid = reshape(prediction_P, (2*nθ, nθ, :))
test_P_grid = reshape(test_P, (2*nθ, nθ, :))
pred_T_grid = reshape(prediction_T, (2*nθ, nθ, :))
test_T_grid = reshape(test_T, (2*nθ, nθ, :))

# Save the results
save("./train/results/$save_name","model_params",model_params,"pred_u_grid",pred_u_grid,
     "test_u_grid",test_u_grid,"pred_v_grid",pred_v_grid,"test_v_grid",test_v_grid,
     "pred_P_grid",pred_P_grid,"test_P_grid",test_P_grid,"pred_T_grid",pred_T_grid,
     "test_T_grid",test_T_grid, compress = true)
println("Results saved. ...")

# Plot the first timestep prediction & ground truth for quick peek
Lat_Lon_Pcolormesh(mesh, pred_u_grid,  1, "./train/plots/baseESN_spectral_pred_u.png")
Lat_Lon_Pcolormesh(mesh, test_u_grid, 1, "./train/plots/baseESN_spectral_test_u.png")
Lat_Lon_Pcolormesh(mesh, pred_v_grid,  1, "./train/plots/baseESN_spectral_pred_v.png")
Lat_Lon_Pcolormesh(mesh, test_v_grid, 1, "./train/plots/baseESN_spectral_test_v.png")
Lat_Lon_Pcolormesh(mesh, pred_P_grid,  1, "./train/plots/baseESN_spectral_pred_P.png")
Lat_Lon_Pcolormesh(mesh, test_P_grid, 1, "./train/plots/baseESN_spectral_test_P.png")
Lat_Lon_Pcolormesh(mesh, pred_T_grid,  1, "./train/plots/baseESN_spectral_pred_T.png")
Lat_Lon_Pcolormesh(mesh, test_T_grid, 1, "./train/plots/baseESN_spectral_test_T.png")
println("Results plotted. ...")

# Countour plot if desired, not currently setup correctly.
#Sigma_Zonal_Mean_Contourf(op_man, "./data/data_plots/spectral_dynamics_contourf")

println("Run complete!")

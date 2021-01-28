using Pkg; Pkg.activate("."); Pkg.instantiate()

using JLD
using JGCM
using ReservoirComputing

include("./primitive_metrics.jl")
using .PrimitivesMetrics

include("../models/base_esn.jl")
using .BaseESN
import PyPlot

using Statistics
using Plots

# Set the filepath for the dataset the model was trained on
dataset_filepath = "./data/datasets/spectral_T21_nd3_500day_100spinup.jld"
#dataset_filepath = "./data/datasets/barotropic_T21_2D_8day.jld"
# Set the name for the trained model and predictions (not path)
model_name = "spectral_T21_nd3_baseESN_res25K_MOD2_norm"

#-- Pick which
#barotropic_evaluate(dataset_filepath, model_name)
spectral_evaluate(dataset_filepath, model_name)


# #-- For manual printing
# model_filepath = "./train/results/$model_name.jld"
# save_path = "./eval/results/$model_name/"
#
# # Load the dataset mesh
# mesh = load(dataset_filepath)["mesh"]
#
# pred_u_grid = load(model_filepath)["pred_u_grid"]
# test_u_grid = load(model_filepath)["test_u_grid"]
# plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=1, animation=false, save_file_name=save_path*"u_t1")
# plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=50, animation=false, save_file_name=save_path*"u_t50")
# plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=100, animation=false, save_file_name=save_path*"u_t100")
# plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=150, animation=false, save_file_name=save_path*"u_t150")

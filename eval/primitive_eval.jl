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

function barotropic_evaluate(dataset_filepath, model_name)
    #-- To be used on saved outputs from ESNs trained on barotropic system
    #-- Inputs: filepath of a trained model
    #-- Outputs: Plots of ________

    # Determine paths from name
    model_filepath = "./train/results/$model_name.jld"
    save_path = "./eval/results/$model_name/"
    mkdir(save_path)

    # Load the dataset mesh
    mesh = load(dataset_filepath)["mesh"]

    # Load the run parameters
    model_params = load(model_filepath)["model_params"]

    # Load the predictions and ground truths
    #-- Shapes are [longitude, latitude, time]
    #-- u is eastward (latitudinal) velocity, v is northward (longitudinal) velocity
    pred_u_grid = load(model_filepath)["pred_u_grid"]
    test_u_grid = load(model_filepath)["test_u_grid"]
    pred_v_grid = load(model_filepath)["pred_v_grid"]
    test_v_grid = load(model_filepath)["test_v_grid"]

    # Calculate full predicted+test statistics
    u = vcat(pred_u_grid, test_u_grid)
    u_mean = mean(u)
    u_std = std(u)
    v = vcat(pred_v_grid, test_v_grid)
    v_mean = mean(v)
    v_std = std(v)

    # Standardize all of the data w/ z-score scaling
    pred_u_grid_s = (pred_u_grid.-u_mean)./u_std
    test_u_grid_s = (test_u_grid.-u_mean)./u_std
    pred_v_grid_s = (pred_v_grid.-v_mean)./v_std
    test_v_grid_s = (test_v_grid.-v_mean)./v_std
    #-- Calculate errors from standardized data
    u_errors = normalized_errors(test_u_grid_s, pred_u_grid_s)
    v_errors = normalized_errors(test_v_grid_s, pred_v_grid_s)

    #-- Plot errors
    barotropic_plot_error(u_errors, v_errors, save_file_name=save_path*"error")

    #-- Create gifs of evolving system
    plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", timestep=1, animation=true, save_file_name=save_path*"u")
    plot_Lat_Lon_mesh(mesh, test_v_grid, pred_v_grid, parameter="Northward Velocity (m/s)", timestep=1, animation=true, save_file_name=save_path*"v")

end

function spectral_evaluate(dataset_filepath, model_name)
    #-- To be used on saved outputs from ESNs trained on spectral_dynamics system
    #-- Inputs: filepath of a trained model
    #-- Outputs: Plots of ________

    # Determine paths from name
    model_filepath = "./train/results/$model_name.jld"
    save_path = "./eval/results/$model_name/"
    mkdir(save_path)

    # Load the dataset mesh
    mesh = load(dataset_filepath)["mesh"]

    # Load the run parameters
    model_params = load(model_filepath)["model_params"]

    # Load the predictions and ground truths
    #-- Shapes are [longitude, latitude, height, time]
    #-- u is eastward velocity, v is northward velocity, P is surface pressure, T is temperature
    pred_u_grid = load(model_filepath)["pred_u_grid"]
    test_u_grid = load(model_filepath)["test_u_grid"]
    pred_v_grid = load(model_filepath)["pred_v_grid"]
    test_v_grid = load(model_filepath)["test_v_grid"]
    pred_P_grid = load(model_filepath)["pred_P_grid"]
    test_P_grid = load(model_filepath)["test_P_grid"]
    pred_T_grid = load(model_filepath)["pred_T_grid"]
    test_T_grid = load(model_filepath)["test_T_grid"]

    # Calculate full predicted+test statistics
    u = vcat(pred_u_grid, test_u_grid)
    u_mean = mean(u)
    u_std = std(u)
    v = vcat(pred_v_grid, test_v_grid)
    v_mean = mean(v)
    v_std = std(v)
    P = vcat(pred_P_grid, test_P_grid)
    P_mean = mean(P)
    P_std = std(P)
    T = vcat(pred_T_grid, test_T_grid)
    T_mean = mean(T)
    T_std = std(T)
    # Standardize all of the data w/ z-score scaling
    pred_u_grid_s = (pred_u_grid.-u_mean)./u_std
    test_u_grid_s = (test_u_grid.-u_mean)./u_std
    pred_v_grid_s = (pred_v_grid.-v_mean)./v_std
    test_v_grid_s = (test_v_grid.-v_mean)./v_std
    pred_P_grid_s = (pred_P_grid.-P_mean)./P_std
    test_P_grid_s = (test_P_grid.-P_mean)./P_std
    pred_T_grid_s = (pred_T_grid.-T_mean)./T_std
    test_T_grid_s = (test_T_grid.-T_mean)./T_std

    # Calculate normalized errors on standardized datasets
    u_errors = normalized_errors(test_u_grid_s, pred_u_grid_s)
    v_errors = normalized_errors(test_v_grid_s, pred_v_grid_s)
    P_errors = normalized_errors(test_P_grid_s, pred_P_grid_s)
    T_errors = normalized_errors(test_T_grid_s, pred_T_grid_s)

    # Plot errors
    # Timestep to end plot for a zoomed in version
    t_end = 250
    spectral_plot_error(u_errors[1:t_end], v_errors[1:t_end], P_errors[1:t_end], T_errors[1:t_end], save_file_name=save_path*"zoom_error")
    # plot all timesteps
    spectral_plot_error(u_errors, v_errors, P_errors, T_errors, save_file_name=save_path*"error")

    # Create gifs of evolving system
    plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=1, animation=true, save_file_name=save_path*"u")
    plot_Lat_Lon_mesh(mesh, test_v_grid, pred_v_grid, parameter="Northward Velocity (m/s)", level=1, timestep=1, animation=true, save_file_name=save_path*"v")
    plot_Lat_Lon_mesh(mesh, test_P_grid, pred_P_grid, parameter="Surface Pressure (Pa)", level=1, timestep=1, animation=true, save_file_name=save_path*"P")
    plot_Lat_Lon_mesh(mesh, test_T_grid, pred_T_grid, parameter="Temperature (K)", level=1, timestep=1, animation=true, save_file_name=save_path*"T")

end


# Set the filepath for the dataset the model was trained on
dataset_filepath = "./data/datasets/spectral_T21_nd3_500day_100spinup.jld"
#dataset_filepath = "./data/datasets/barotropic_T21_2D_8day.jld"
# Set the name for the trained model and predictions (not path)
model_name = "spectral_T21_nd3_baseESN_res25K_MOD2_norm"

#-- Pick which
#barotropic_evaluate(dataset_filepath, model_name)
#spectral_evaluate(dataset_filepath, model_name)


# -- For manual printing
model_filepath = "./train/results/$model_name.jld"
save_path = "./eval/results/$model_name/"

# Load the dataset mesh
mesh = load(dataset_filepath)["mesh"]

pred_u_grid = load(model_filepath)["pred_u_grid"]
test_u_grid = load(model_filepath)["test_u_grid"]
plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=1, animation=false, save_file_name=save_path*"u_t1")
plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=50, animation=false, save_file_name=save_path*"u_t50")
plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=100, animation=false, save_file_name=save_path*"u_t100")
plot_Lat_Lon_mesh(mesh, test_u_grid, pred_u_grid, parameter="Eastward Velocity (m/s)", level=1, timestep=150, animation=false, save_file_name=save_path*"u_t150")

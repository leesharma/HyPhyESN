module PrimitivesMetrics
  export barotropic_evaluate, spectral_evaluate, plot_Lat_Lon_mesh, l2_dist, normalized_errors, time_horizon, barotropic_plot_error, spectral_plot_error

  using JLD
  using Plots
  using Statistics
  using JGCM
  using PyPlot
  using ReservoirComputing

  import PyPlot

  include("../models/base_esn.jl")
  using .BaseESN

  # largest Lyapunov exponent (used for timescale)
  lyapunov_max = 0.934

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


  function plot_Lat_Lon_mesh(mesh::Spectral_Spherical_Mesh, in_true_grid, in_pred_grid; level::Int64=-1, parameter::String, timestep, animation=false, save_file_name::String = "None")
      """
      Returns a stacked plot showing groundtruth and prediction componants.
      2D slice through <height> plane, shows latitudinal and longitudinal values.
      """
    # Clear any existing plots. Mostly for animation.
    PyPlot.clf()

    # Generate plot mesh
    λc, θc = mesh.λc, mesh.θc
    nλ, nθ = length(λc), length(θc)
    λc_deg, θc_deg = λc*180/pi, θc*180/pi

    X,Y = repeat(λc_deg, 1, nθ), repeat(θc_deg, 1, nλ)'

    fig, (ax1, ax2, ax3) = PyPlot.subplots(1, 3, figsize=(12,4))
    fig.suptitle(parameter)

    # Generate figure
    fig, (ax1, ax3, ax2, ax4) = PyPlot.subplots(2, 2, figsize=(12,8))

    # Get global maxima and minima for colorbars over all timesteps
    cmax = max(maximum(in_true_grid), maximum(in_pred_grid))
    cmin = min(minimum(in_true_grid), minimum(in_pred_grid))
    error_max = maximum(abs.(in_true_grid -  in_pred_grid))
    error_min = 0
    #perc_error_max = 100
    #perc_error_max = maximum((abs.(in_true_grid -  in_pred_grid)./ abs.(in_true_grid)).*100) # If you want to dynamically set upper bound
    #perc_error_min = minimum((abs.(in_true_grid -  in_pred_grid)./ abs.(in_true_grid)).*100) # If you want to dynamically set lower bound
    #perc_error_min = 10e-3

    # Structure data based on input type
    if level != -1  # spectral_dynamics data
        y_true_grid = in_true_grid[:,:,level,timestep]
        y_pred_grid = in_pred_grid[:,:,level,timestep]
        # Generate Title & Subtitle
        timestamp = round(timestep*1200/3600, digits=2) # Assumes integrator using default timestep
        title = fig.suptitle(parameter*"\n$timestamp hours")
    else  # barotropic data
        y_true_grid = in_true_grid[:,:,timestep]
        y_pred_grid = in_pred_grid[:,:,timestep]
        # Generate Title & Subtitle
        timestamp = timestep*1800/3600 # Assumes integrator using default timestep
        title = fig.suptitle(parameter*"\n$timestamp hours")
    end

    # Calculate relevant quantities
    abs_error = abs.(y_true_grid -  y_pred_grid)
    perc_error = ((abs_error)./ abs.(y_true_grid)).*100

    #- Get local max for dynamic colorbars
    # perc_error_max = maximum(perc_error)
    # perc_error_min = minimum(perc_error)
    # cmax = max(maximum(y_true_grid),maximum(y_pred_grid))
    # cmin = min(minimum(y_true_grid), minimum(y_pred_grid))

    # Generate plots
    im1 = ax1.pcolormesh(X, Y, y_true_grid, shading= "gouraud", cmap="viridis", vmax = cmax, vmin=cmin)
    im2 = ax2.pcolormesh(X, Y, y_pred_grid, shading= "gouraud", cmap="viridis", vmax = cmax, vmin=cmin)
    im3 = ax3.pcolormesh(X, Y, abs_error, shading= "gouraud", cmap="magma", vmax = error_max, vmin=error_min)
    im4 = ax4.pcolormesh(X, Y, perc_error, shading= "gouraud", cmap="magma", norm = PyPlot.matplotlib.colors.LogNorm(vmin=0.001, vmax=100))

    # Set colorbars
    cb1 = fig.colorbar(im1, ax = ax1)
    cb2 = fig.colorbar(im2, ax = ax2)
    cb3 = fig.colorbar(im3, ax = ax3)
    cb4 = fig.colorbar(im4, ax = ax4, ticks=[0.01, 0.1, 1, 10, 100])
    cb4.ax.set_yticklabels(["< 0.01%","0.1%", "1%", "10%", "> 100%"])

    # Title subplots
    ax1.set_title("Test Data")
    ax2.set_title("Predicted Data")
    ax3.set_title("Absolute Error")
    ax4.set_title("Percent Error (log scale)")

    # Set axis labels
    ax1.set_ylabel("Latitude")
    ax3.set_ylabel("Latitude")
    ax3.set_xlabel("Longitude")
    ax4.set_xlabel("Longitude")

    function animate(iter)
        if level != -1  # spectral_dynamics data
            offset = 3  # Number of timesteps to skip ahead with each frame
            y_true_grid = in_true_grid[:,:,level,offset*iter+1]
            y_pred_grid = in_pred_grid[:,:,level,offset*iter+1]
            # Update Title timestamp
            timestamp = round(offset*iter*1200/3600, digits = 2) # Assumes integrator using default timestep
            title.set_text(parameter*"\n$timestamp hours")
        else  # barotropic data
            y_true_grid = in_true_grid[:,:,iter+1]
            y_pred_grid = in_pred_grid[:,:,iter+1]
            # Update title timestamp
            timestamp = iter*1800/3600 # Assumes integrator using default timestep
            title.set_text(parameter*"\n$timestamp hours")
        end
        abs_error = abs.(y_true_grid -  y_pred_grid)
        perc_error = ((abs_error)./ abs.(y_true_grid)).*100
        #-- For changing colobars, uncomment these
        # cmax = max(maximum(y_true_grid),maximum(y_pred_grid))
        # cmin = min(minimum(y_true_grid), minimum(y_pred_grid))
        # cb1.remove()
        # cb2.remove()
        # cb3.remove()
        # cb4.remove()
        # im1 = ax1.pcolormesh(X, Y, y_true_grid, shading= "gouraud", cmap="viridis", vmax = cmax, vmin=cmin)
        # im2 = ax2.pcolormesh(X, Y, y_pred_grid, shading= "gouraud", cmap="viridis", vmax = cmax, vmin=cmin)
        # im3 = ax3.pcolormesh(X, Y, abs_error, shading= "gouraud", cmap="viridis")
        # im4 = ax4.pcolormesh(X, Y, perc_error, shading= "gouraud", cmap="viridis", norm = PyPlot.matplotlib.colors.LogNorm(vmin=perc_error_min, vmax=perc_error_max))  # matplotlib[:colors][:LogNorm]
        # # Set colorbars
        # cb1 = fig.colorbar(im1, ax = ax1)
        # cb2 = fig.colorbar(im2, ax = ax2)
        # cb3 = fig.colorbar(im3, ax = ax3)
        # cb4 = fig.colorbar(im4, ax = ax4)
        # # Title subplots
        # ax1.set_title("Test Data")
        # ax2.set_title("Predicted Data")
        # ax3.set_title("Absolute Error")
        # ax4.set_title("Percent Error (log scale)")
        # # Set axis labels
        # ax1.set_ylabel("Latitude")
        # ax3.set_ylabel("Latitude")
        # ax3.set_xlabel("Longitude")
        # ax4.set_xlabel("Longitude")
        #-- For constant colorbars, use below
        im1.set_array(y_true_grid)
        im2.set_array(y_pred_grid)
        im3.set_array(abs_error)
        im4.set_array(perc_error)
        return im1,im2,im3,im4,fig,title
    end

    if animation
        #If animating, do so and save
        anim = PyPlot.matplotlib.animation.FuncAnimation(fig,animate,frames=90,interval=100)
        anim.save(save_file_name*".gif")
    elseif save_file_name != "None"
        # Otherwise, save if desired
        fig.savefig(save_file_name*".png")
    end
  end

  function l2_dist(arr)
      #-- L2 matrix norm, assuming a temporal component in last dim.
      #-- Returns a 1D matrix with length of temporal shape
      # Reshape into m x n, where m is all spatial components and n is temporal
      arr = reshape(arr,(:,size(arr)[end]))
      # Square elements, sum over all spatial elements, sqrt remaining elements
      dist = sqrt.(sum(arr.^2,dims=1))
      return dist
  end

  """
    normalized_error(y, y_pred)

  Returns the L2 error normalized by the time-average of y (as in Doan, 2019):
    ||y-y_pred|| / sqrt( < ||y||^2 > )

  Returns a 1D array.
  """
  function normalized_errors(y, y_pred)
    errors = l2_dist(y-y_pred)                    # L2 error for each pred
    # Flatten spatial components of y, keep temporal dim
    y_flat = reshape(y,(:,size(y)[end]))
    scaling_factor = sqrt(mean(sum(y_flat.^2, dims=1)))  # time-avg magnitude of y

    normalized_errors = vec(errors./scaling_factor)
  end

  """
    time_horizon(y, y_pred)

  Returns the time over which the function remains in the specified error bounds.
  """
  function time_horizon(y, y_pred; E_max=0.4, dt=1)
    good_points = collect(Iterators.takewhile(<(E_max), normalized_errors(y,y_pred)))
    timesteps = length(good_points)
    timesteps*dt
  end

  """
    plot_error(y, y_pred)

  Plots the instantaneous error over time. Shows a horizontal marker for max
  error level (for the time horizon calculation).
  """
  function barotropic_plot_error(u_error, v_error; E_max=0.4, dt=1, save_file_name)
    n = size(u_error,1)
    #x = range(1, dt*n, length=n)  # scale by largest Lyapunov exponent

    plt = Plots.plot(title="Barotropic System Prediction Error", legend=:topleft)
    plot!(plt, u_error, label="Eastward Velocity")
    plot!(plt, v_error, label="Northward Velocity")
    plot!(plt, repeat([E_max],n), label="Max error target", linestyle=:dot)
    ylabel!(plt, "Normalized Error")
    xlabel!(plt, "Time Steps (1 step = 30 minutes)")
    Plots.savefig(save_file_name)
  end

  function spectral_plot_error(u_error, v_error, P_error, T_error; E_max=0.4, dt=1, save_file_name)
    n = size(u_error,1)
    #x = range(1, dt*n, length=n)  # scale by largest Lyapunov exponent

    plt = Plots.plot(title="Spectral Dynamical Core Prediction Error", legend=:topleft)
    plot!(plt, u_error, label="Eastward Velocity")
    plot!(plt, v_error, label="Northward Velocity")
    plot!(plt, P_error, label="Surface Pressure")
    plot!(plt, T_error, label="Temperature")
    plot!(plt, repeat([E_max],n), label="Max error target", linestyle=:dot)
    ylabel!(plt, "Normalized Error")
    xlabel!(plt, "Time Steps (1 step = 20 minutes)")

    Plots.savefig(save_file_name)
  end

end # End

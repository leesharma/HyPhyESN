module Metrics
  export root_mean_squared_errors

  using Plots
  using Statistics: mean


  # largest Lyapunov exponent (used for timescale)
  lyapunov_max = 0.934


  # Metrics

  """
      root_mean_squared_errors(y, y_pred)

  Returns a list of RMSE values for each predicted feature.
  """
  function root_mean_squared_errors(y, y_pred)
    err = y-y_pred
    dims, n = size(y)

    sqrt.(1/n * [err[i]'*err[i] for i in 1:dims])
  end

  """
    normalized_error(y, y_pred)

  Returns the L2 error normalized by the time-average of y (as in Doan, 2019):
    ||y-y_pred|| / sqrt( < ||y||^2 > )

  Returns a 1D array.
  """
  function normalized_errors(y, y_pred)
    l2_dist(vec) = sqrt(sum(vec.^2))

    errors = l2_dist.(eachcol(y-y_pred))              # L2 error for each pred
    scaling_factor = sqrt(mean(sum.(eachcol(y.^2))))  # time-avg magnitude of y

    normalized_errors = errors./scaling_factor
  end

  """
    time_horizon(y, y_pred)

  Returns the time over which the function remains in the specified error bounds.

  By default, this is scaled by the max Lyapunov exponent.
  """
  function time_horizon(y, y_pred; E_max=0.4, timescale=lyapunov_max, dt=1)
    good_points = collect(Iterators.takewhile(<(E_max), normalized_errors(y,y_pred)))
    timesteps = length(good_points)
    timesteps*dt*timescale
  end


  # Plots (for Fun and Profit)

  """
      plot_predictions_animated(y, y_pred, output_file; kwargs...)

  Saves an animated gif of X, Y, and Z predictions and groundtruth evolving over time.
  """
  function plot_predictions_animated(y, y_pred, output_file; labels=["X","Y","Z"])
    n = size(y)[2]
    anim = @animate for i in 1:n
      plot_predictions(y[:,1:i], y_pred[:,1:i], labels=labels, xmax=n)
    end every 10
    gif(anim, output_file, fps=15)
  end

  """
      plot_predictions(y, y_pred; kwargs...)

  Returns a stacked plot showing the X, Y, and Z groundtruth and prediction componants.
  """
  function plot_predictions(y, y_pred; labels=["X","Y","Z"], xmax=size(y)[2], dt=1)
    dims, n = size(y)
    x = range(1, dt*n, length=n) * lyapunov_max  # scale by largest Lyapunov exponent

    xlims = (0,x[end])
    ylims = [(-18,18),(-22,26),(3,47)]

    plt = plot(layout=(dims,1))
    plot!(plt, title="Lorenz Attractor Predictions", subplot=1)
    for feature_idx in 1:dims
      plot!(plt, x, y[feature_idx,:], lab="\$$(labels[feature_idx])\$", xlims=xlims, ylims=ylims[feature_idx], subplot=feature_idx)
      plot!(plt, x, y_pred[feature_idx,:], lab="\$$(labels[feature_idx])_{pred}\$", xlims=xlims, ylims=ylims[feature_idx], subplot=feature_idx)
      ylabel!(plt, labels[feature_idx], subplot=feature_idx)
    end
    xlabel!(plt, "\$\\lambda_{max}t\$", subplot=dims)
    plt
  end

  """
    plot_error(y, y_pred)

  Plots the instantaneous error over time. Shows a horizontal marker for max
  error level (for the time horizon calculation).
  """
  function plot_error(y, y_pred; labels=["X","Y","Z"], errors=normalized_errors(y,y_pred), E_max=0.4, dt=1)
    n = size(errors,1)
    x = range(1, dt*n, length=n) * lyapunov_max  # scale by largest Lyapunov exponent

    plt = plot()
    plot!(plt, title="Lorenz Attractor Prediction Error")
    plot!(plt, x, errors, label="Prediction error")
    plot!(plt, x, repeat([E_max],n), label="Max error target")
    ylabel!(plt, "Normalized Error")
    xlabel!(plt, "\$\\lambda_{max}t\$")
    plt
  end

  """
    plot_avg_error(y, y_pred)

  Plots the average error over time.
  """
  function plot_avg_error(y, y_pred; labels=["X","Y","Z"], errors=normalized_errors(y,y_pred), E_max=0.4, dt=1)
    time_avg(data) = [sum(data[1:i])/i for i in 1:length(data)]

    n = size(errors,1)
    x = range(1, dt*n, length=n) * lyapunov_max  # scale by largest Lyapunov exponent

    plt = plot()
    plot!(plt, title="Lorenz Attractor Time-Average Prediction Error")
    plot!(plt, x, time_avg(errors), legend=false)
    ylabel!(plt, "Average Normalized Error")
    xlabel!(plt, "\$\\lambda_{max}t\$")
    plt
  end
end # module Metrics

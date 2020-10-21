module Metrics
  export root_mean_squared_errors

  using Plots


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



  # Plots for Fun and Profit

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
  function plot_predictions(y, y_pred; labels=["X","Y","Z"], xmax=size(y)[2])
    dims = length(labels)
    xlims = (0,xmax)
    ylims = [(-18,18),(-22,26),(3,47)]

    plt = plot(layout=(dims,1))
    plt = plot!(title="Lorenz Attractor Predictions", subplot=1)
    for feature_idx in 1:dims
      plt = plot!(y[feature_idx,:], lab="\$$(labels[feature_idx])\$", xlims=xlims, ylims=ylims[feature_idx], subplot=feature_idx)
      plt = plot!(y_pred[feature_idx,:], lab="\$$(labels[feature_idx])_{pred}\$", xlims=xlims, ylims=ylims[feature_idx], subplot=feature_idx)
      plt = ylabel!(labels[feature_idx], subplot=feature_idx)
    end
    plt = xlabel!("timesteps", subplot=dims)
    plt
  end

  """
      plot_errors(y, y_pred; kwargs...)

  Plots absolute prediction error of each component over time.
  """
  function plot_errors(y, y_pred; labels=["X","Y","Z"])
    dims = length(labels)
    errors = abs.(y-y_pred)

    plt = plot(layout=(dims,1))
    plt = plot!(title="Lorenz Attractor Prediction Error", subplot=1)
    for feature_idx in 1:dims
      plt = plot!(errors[feature_idx,:], legend=false, subplot=feature_idx)
      plt = ylabel!(labels[feature_idx], subplot=feature_idx)
    end
    plt = xlabel!("timesteps", subplot=dims)
    plt
  end
end # module Metrics

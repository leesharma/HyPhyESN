include("../data/barotropic.jl")
using .BarotropicData

include("../models/base_esn.jl")
using .BaseESN

using JGCM
using JLD


###############################################################################
#-- Training parameters
dataset_filepath = "./data/datasets/barotropic_T21_8day.jld"
approx_res_size = 8000  # NOTE this must be larger than size(train_u)[1]

# Load some parameters from the dataset
end_time = load(dataset_filepath)["end_time"]
Δt = load(dataset_filepath)["Δt"]

# ESN training parameters. Currently throws away first 1/4 of data, trains with next 1/2, tests on final 1/4.
train_len = floor(Int64, (end_time/Δt)/2)
predict_len = ceil(Int64, (end_time/Δt)/4)
shift = floor(Int64, (end_time/Δt)/4)
###############################################################################


# Generate the data
train_u, test_u, train_v, test_v, mesh = BarotropicData.train_test(dataset_filepath,
                                                                   train_len = train_len,
                                                                   predict_len = predict_len,
                                                                   shift = shift)
nθ = mesh.nθ

train_data = cat(train_u, train_v, dims=1)

# Initialize ESN, then train, & predict
esn = BaseESN.esn(train_data, approx_res_size)
W_out = BaseESN.train(esn)
prediction = BaseESN.predict(esn, predict_len, W_out)

# Separate u & v from prediction array, reshape them to grid shape
prediction_u = prediction[1:Int64(size(prediction)[1]/2),:]
prediction_v = prediction[Int64(size(prediction)[1]/2+1):end,:]
pred_u_grid = reshape(prediction_u, (2*nθ, nθ, :))
test_u_grid = reshape(test_u, (2*nθ, nθ, :))
pred_v_grid = reshape(prediction_v, (2*nθ, nθ, :))
test_v_grid = reshape(test_v, (2*nθ, nθ, :))

# Plot the first timestep prediction & ground truth for quick peek
Lat_Lon_Pcolormesh(mesh, pred_u_grid,  1, "./train/plots/baseESN_barotropic_pred_u.png")
Lat_Lon_Pcolormesh(mesh, test_u_grid, 1, "./train/plots/baseESN_barotropic_test_u.png")
Lat_Lon_Pcolormesh(mesh, pred_v_grid,  1, "./train/plots/baseESN_barotropic_pred_v.png")
Lat_Lon_Pcolormesh(mesh, test_v_grid, 1, "./train/plots/baseESN_barotropic_test_v.png")

println("Completed!")

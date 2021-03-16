using JLD

function hdf5_to_JLD(dataset_directory, save_directory)

    """
    This takes an .hdf5 file prepared by CFSR_Preprocess_1.py and converts it
    to a JLD file for use in our BaseESN.
    """

    file = jldopen(dataset_directory)
    dictionary = read(file)

    save(save_directory,"temporal_grid_u",dictionary["temporal_grid_u"],"temporal_grid_v",dictionary["temporal_grid_v"],"temporal_grid_P",dictionary["temporal_grid_P"],"temporal_grid_T",dictionary["temporal_grid_T"],compress = true)
    close(file)
end

function CFS_reforecast_combine(CFS_dataset_directory, CFSR_dataset_directory, save_directory)
    """
    This takes a processed CFS reforecast dataset and merges it with a CFSR dataset
    in the same format as baseESN predictions. Here, `pred_grid` variables are
    entries from the CFS reforecast, and `test_grid` variables are real world
    data from the associated CFSR data.
    """

    #-- Load CFSR (Reanalysis) data in as test data.
    # NOTE: Need to extract portion of data you're interested in.

    # Set the ratio of data used for train/predict (usually whatever's used in the ESN you're comparing to)
    train_ratio = 3/4
    predict_ratio = 1/4

    #-- For each variable of interest, load it, cut out the relevant portion, clear memory, reshape, and take only every 6 hours
    # This is done iteratively to minimize memory usage.

    # NOTE: Only taking the 1st altitude for all variables here.
    # To take all of them, change temporal_grid_X[:,:,1, ...] to [:,:,:,...]

    # Load the data
    temporal_grid_u = load(CFSR_dataset_directory)["temporal_grid_u"]

    # Grab dataset params
    total_time = size(temporal_grid_u)[4]
    # Determine indices to split on
    train_len = floor(Int64, train_ratio*total_time)
    predict_len = floor(Int64, predict_ratio*total_time)

    #-- u
    test_u_grid = temporal_grid_u[:,:,1, train_len:train_len+predict_len-1]
    temporal_grid_u = 0
    test_u_grid = reshape(test_u_grid,(size(test_u_grid)[1], size(test_u_grid)[2], 1, size(test_u_grid)[3]))
    test_u_grid = test_u_grid[:,:,:,6:6:end]

    #-- v
    temporal_grid_v = load(CFSR_dataset_directory)["temporal_grid_v"]
    test_v_grid = temporal_grid_v[:,:,1, train_len:train_len+predict_len-1]
    temporal_grid_v = 0
    test_v_grid = reshape(test_v_grid,(size(test_v_grid)[1], size(test_v_grid)[2], 1, size(test_v_grid)[3]))
    test_v_grid = test_v_grid[:,:,:,6:6:end]

    #-- P
    temporal_grid_P = load(CFSR_dataset_directory)["temporal_grid_P"]
    test_P_grid = temporal_grid_P[:,:,1, train_len:train_len+predict_len-1]
    temporal_grid_P = 0
    test_P_grid = reshape(test_P_grid,(size(test_P_grid)[1], size(test_P_grid)[2], 1, size(test_P_grid)[3]))
    test_P_grid = test_P_grid[:,:,:,6:6:end]

    #-- T
    temporal_grid_T = load(CFSR_dataset_directory)["temporal_grid_T"]
    test_T_grid = temporal_grid_T[:,:,1, train_len:train_len+predict_len-1]
    temporal_grid_T = 0
    test_T_grid = reshape(test_T_grid, (size(test_T_grid)[1], size(test_T_grid)[2], 1, size(test_T_grid)[3]))
    test_T_grid = test_T_grid[:,:,:,6:6:end]

    #-- Load CFS Reforecast data in as prediction data
    pred_u_grid = load(CFS_dataset_directory)["temporal_grid_u"]
    pred_v_grid = load(CFS_dataset_directory)["temporal_grid_v"]
    pred_P_grid = load(CFS_dataset_directory)["temporal_grid_P"]
    pred_T_grid = load(CFS_dataset_directory)["temporal_grid_T"]

    # Cut test data to same length as prediction data
    test_u_grid = test_u_grid[:,:,:,1:size(pred_u_grid)[4]]
    test_v_grid = test_v_grid[:,:,:,1:size(pred_v_grid)[4]]
    test_P_grid = test_P_grid[:,:,:,1:size(pred_P_grid)[4]]
    test_T_grid = test_T_grid[:,:,:,1:size(pred_T_grid)[4]]

    # Save the results
    save(save_directory,"pred_u_grid",pred_u_grid,"test_u_grid",test_u_grid,
    "pred_v_grid",pred_v_grid,"test_v_grid",test_v_grid,"pred_P_grid",pred_P_grid,
    "test_P_grid",test_P_grid,"pred_T_grid",pred_T_grid,"test_T_grid",test_T_grid,
    compress = true)

    println("Results saved. ...")


end


# -- For hdf5_to_JLD(...) processing, use the below.
# dataset_directory = "E:/HyPhyESN_Datasets/CFS_Reforecast/45-day/HighPriority/CFS_Reforecast_45day_2006070112.hdf5"
# save_directory = "E:/HyPhyESN_Datasets/CFS_Reforecast/45-day/HighPriority/CFS_Reforecast_45day_2006070112.jld"
#
# hdf5_to_JLD(dataset_directory, save_directory)

###############################################################################

#-- For CFS_reforecast_combine(...) processing, use the below

CFS_dataset_directory = "E:/HyPhyESN_Datasets/CFS_Reforecast/45-day/HighPriority/CFS_Reforecast_45day_2006070112.jld"
CFSR_dataset_directory = "E:/HyPhyESN_Datasets/CFSR/T62/CFSR_T62_2year_3height.jld"
save_directory = "E:/HyPhyESN_Datasets/CFS_Reforecast/45-day/HighPriority/CFS_CFSR_combined_45day_2006070112.jld"

CFS_reforecast_combine(CFS_dataset_directory, CFSR_dataset_directory, save_directory)

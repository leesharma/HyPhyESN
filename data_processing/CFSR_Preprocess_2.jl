using JLD

function hdf5_to_JLD(dataset_directory, save_directory)

    """
    This takes an .hdf5 file prepared by CFSR_Preprocess_1.py and converts it
    to a JLD file for use in our BaseESN.
    """

    file = jldopen(dataset_directory)
    dictionary = read(file)

    save(save_directory,"temporal_grid_u",dictionary["temporal_grid_u"],
         "temporal_grid_v",dictionary["temporal_grid_v"],"temporal_grid_P",dictionary["temporal_grid_P"],
         "temporal_grid_T",dictionary["temporal_grid_T"],compress = true)
    close(file)
end


dataset_directory = "E:/HyPhyESN_Datasets/CFSR/T62/CFSR_2year.hdf5"
save_directory = "E:/HyPhyESN_Datasets/CFSR/T62/CFSR_2year.jl"

hdf5_to_JLD(dataset_directory, save_directory)

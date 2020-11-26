## Generating, Saving, & Training on Data

### Generating & Saving Data
`barotropic.jl` and `spectral_dynamics.jl` contain a simplified and more complex dynamical core, respectively, for weather forecasting.

Each contains a function, `generate_data` that will generate and save a dataset with defined parameters to the `datasets` directory.

Parameters for data generation should be changed in the generator functions themselves, as they do not currently take arguments.

### Training on Data
`barotropic.jl` and `spectral_dynamics.jl` each contain a function, `train_test`, which will take a saved dataset and split it into training and test sets with the correct format for our ESNs. The `train_test` function accepts the `dataset_filepath`, `train_len`, and `predict_len` as arguments. `barotropic.jl`'s `train_test` function also accepts a `shift` argument.

`train_len` and `predict_len` specify the number of timesteps in the training and test sets, respectively. `shift` specifies the number of timesteps from the beginning of the set to be thrown away. `spectral_dynamics.jl` does not contain a `shift` argument because the solver already throws away a number of timesteps during solving (known as `spinup_day`). This number can be modified when generating the dataset.

### In Practice
Scripts have already been created in the `HyPhyESN/train` directory which will let you specify run parameters and automatically retrieve the data, apply `train_test`, and train the ESN.

Datasets are not being uploaded to github. Datasets can be downloaded ([here](https://drive.google.com/drive/folders/1fVd7ErWMpxY1Bez2_uQKyq0fXLktF27O?usp=sharing)) (NOTE: you must login with UMD account), and should be placed in `HyPhyESN/data/datasets` directory.

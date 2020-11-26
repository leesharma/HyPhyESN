module SpectralData

    export train_test, generate_dataset

    using JGCM
    using JLD

    function Atmos_Spectral_Dynamics_Main(physics_params::Dict{String, Float64}, num_fourier::Int64 = 42, nθ::Int64 = 64, nd::Int64 = 20, end_day::Int64 = 1200, spinup_day::Int64 = 200)
      # the decay of a sinusoidal disturbance to a zonally symmetric flow
      # that resembles that found in the upper troposphere in Northern winter. (this description might be wrong?)
      name = "Spectral_Dynamics"
      #num_fourier, nθ, nd = 42, 64, 20  # T42 defaults
      #num_fourier, nθ, nd = 21, 32, 20  # T21 defaults

      num_spherical = num_fourier + 1
      nλ = 2nθ

      radius = 6371000.0
      omega = 7.292e-5   # Do not change without updating hybridM_spectral()
      sea_level_ps_ref = 1.0e5  # Do not change without updating hybridM_spectral()
      init_t = 264.0

      # Initialize mesh
      mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
      θc, λc = mesh.θc,  mesh.λc
      cosθ, sinθ = mesh.cosθ, mesh.sinθ

      # Do not change the vert_coord without updating the hybridM_spectral() function
      vert_coord = Vert_Coordinate(nλ, nθ, nd, "even_sigma", "simmons_and_burridge", "second_centered_wts", sea_level_ps_ref)
      # Initialize atmo_data. Do not change without updating hybridM_spectral()
      do_mass_correction = true
      do_energy_correction = true
      do_water_correction = false

      use_virtual_temperature = false
      atmo_data = Atmo_Data(name, nλ, nθ, nd, do_mass_correction, do_energy_correction, do_water_correction, use_virtual_temperature, sinθ, radius,  omega)

      # Initialize integrator. Do not change without updating hybridM_spectral()
      damping_order = 4
      damping_coef = 1.15741e-4
      robert_coef  = 0.04

      implicit_coef = 0.5
      day_to_sec = 86400
      start_time = 0
      end_time = end_day*day_to_sec  #
      Δt = 1200
      init_step = true

      integrator = Filtered_Leapfrog(robert_coef,
      damping_order, damping_coef, mesh.laplacian_eig,
      implicit_coef, Δt, init_step, start_time, end_time)

      ps_ref = sea_level_ps_ref
      t_ref = fill(300.0, nd)
      wave_numbers = mesh.wave_numbers
      semi_implicit = Semi_Implicit_Solver(vert_coord, atmo_data,
      integrator, ps_ref, t_ref, wave_numbers)

      # Data Visualization
      op_man= Output_Manager(mesh, vert_coord, start_time, end_time, spinup_day)

      # Initialize data
      dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd)

      NT = Int64(end_time / Δt)

      # Initialize a var to store u, v, P, T for each time step
      temporal_grid_dims = (nλ, nθ, nd, NT)
      surface_temporal_grid_dims = (nλ, nθ, 1, NT)
      temporal_grid_u = zeros(Float64, temporal_grid_dims)
      temporal_grid_v = zeros(Float64, temporal_grid_dims)
      temporal_grid_P = zeros(Float64, surface_temporal_grid_dims)
      temporal_grid_T = zeros(Float64, temporal_grid_dims)

      Get_Topography!(dyn_data.grid_geopots)

      Spectral_Initialize_Fields!(mesh, atmo_data, vert_coord, sea_level_ps_ref, init_t,
      dyn_data.grid_geopots, dyn_data)

      Atmosphere_Update!(mesh, atmo_data, vert_coord, semi_implicit, dyn_data, physics_params)
      Update_Init_Step!(semi_implicit)
      integrator.time += Δt
      Update_Output!(op_man, dyn_data, integrator.time)

      temporal_grid_u[:,:,:,1] = dyn_data.grid_u_c
      temporal_grid_v[:,:,:,1] = dyn_data.grid_v_c
      temporal_grid_P[:,:,1,1] = dyn_data.grid_ps_c # Surface pressure
      temporal_grid_T[:,:,:,1] = dyn_data.grid_t_c

      for i = 2:NT

        Atmosphere_Update!(mesh, atmo_data, vert_coord, semi_implicit, dyn_data, physics_params)

        integrator.time += Δt
        #@info integrator.time

        Update_Output!(op_man, dyn_data, integrator.time)

        if (integrator.time%day_to_sec == 0)
          @info "Day: ", div(integrator.time,day_to_sec), " Max |U|,|V|,|P|,|T| : ", maximum(abs.(dyn_data.grid_u_c)), maximum(abs.(dyn_data.grid_v_c)), maximum(dyn_data.grid_p_full), maximum(dyn_data.grid_t_c)
        end

        temporal_grid_u[:,:,:,i] = dyn_data.grid_u_c
        temporal_grid_v[:,:,:,i] = dyn_data.grid_v_c
        temporal_grid_P[:,:,:,i] = dyn_data.grid_ps_c
        temporal_grid_T[:,:,:,i] = dyn_data.grid_t_c

      end

      return op_man, mesh, temporal_grid_u, temporal_grid_v, temporal_grid_P, temporal_grid_T

    end

    function train_test(dataset_filepath,
                        train_len,
                        predict_len)
        #-- Takes a dataset & training params as input, outputs training/test sets for ESN
        # Load data
        op_man = load(dataset_filepath)["op_man"]
        temporal_grid_u = load(dataset_filepath)["temporal_grid_u"]
        temporal_grid_v = load(dataset_filepath)["temporal_grid_v"]
        temporal_grid_P = load(dataset_filepath)["temporal_grid_P"]
        temporal_grid_T = load(dataset_filepath)["temporal_grid_T"]

        end_day = op_man.end_time
        spinup_day = op_man.spinup_day

        # Convert day parameters to seconds, then divide by time step to get array indices
        day_to_sec = 86400
        Δt = 1200
        shift = floor(Int64, (spinup_day*day_to_sec)/Δt)
        train_len = floor(Int64, (train_len*day_to_sec)/Δt)
        predict_len = floor(Int64, (predict_len*day_to_sec)/Δt)

        # Grab first vertical component (others aren't solved), flatten the remaining spatial components
        # Leaves us a 2D array of spatial solutions & time step
        data_u = reshape(temporal_grid_u[:,:,1,:], (:,size(temporal_grid_u)[4]))
        data_v = reshape(temporal_grid_v[:,:,1,:], (:,size(temporal_grid_v)[4]))
        data_P = reshape(temporal_grid_P[:,:,1,:], (:,size(temporal_grid_P)[4]))
        data_T = reshape(temporal_grid_T[:,:,1,:], (:,size(temporal_grid_T)[4]))
        train_u = data_u[:, shift:shift+train_len-1]
        train_v = data_v[:, shift:shift+train_len-1]
        train_P = data_P[:, shift:shift+train_len-1]
        train_T = data_T[:, shift:shift+train_len-1]
        test_u = data_u[:, shift+train_len:shift+train_len+predict_len-1]
        test_v = data_v[:, shift+train_len:shift+train_len+predict_len-1]
        test_P = data_P[:, shift+train_len:shift+train_len+predict_len-1]
        test_T = data_T[:, shift+train_len:shift+train_len+predict_len-1]

        return  op_man, train_u, train_v, train_P, train_T, test_u, test_v, test_P, test_T
    end

    function generate_dataset()
        #-- Generates and saves grids for u, v, P, T for later use.

        ###############################################################################
        # Name of dataset folder to save
        file_name = "spectral_T21_600day_200spinup.jld"

        # Data generation parameters
        # T21 grid
        num_fourier = 21  # Fourier wave number truncation
        nθ = 32  # Latitudinal grid size
        nd = 20  # Vertical slices, doesn't actually matter here, it just solves nd = 1

        # T42 grid
        #num_fourier = 42
        #nθ = 64
        #nd = 20

        # CFS standard spectral grid resolution, T126L64
        #num_fourier = 126
        #nθ = 190
        #nd = 64

        end_day = 600  # Default 1200
        spinup_day = 200  # Default 200. How long to run the solver before taking data.
        ################################################################################

        #include("HS.jl")  # For adding in parameterizations. Not needed right now.
        # Build param dict
        physics_params = Dict{String,Float64}("σ_b"=>0.7, "k_f" => 1.0, "k_a" => 1.0/40.0, "k_s" => 1.0/4.0, "ΔT_y" => 60.0, "Δθ_z" => 10.0)
        # Generate the data
        op_man, mesh, temporal_grid_u, temporal_grid_v, temporal_grid_P, temporal_grid_T = Atmos_Spectral_Dynamics_Main(physics_params, num_fourier, nθ, nd, end_day, spinup_day)
        #Finalize_Output!(op_man, "HS_OpM.dat", "HS_mean.dat")  # To implement parameterizations

        # Save the data
        save("./data/datasets/$file_name","op_man",op_man,"mesh",mesh,"temporal_grid_u",temporal_grid_u,"temporal_grid_v",temporal_grid_v,"temporal_grid_P",temporal_grid_P,"temporal_grid_T",temporal_grid_T, compress = true)
        println("Data saved.")
    end

end

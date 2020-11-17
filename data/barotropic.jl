module BarotropicData

    export train_test

    using JGCM

    # Default initial conditions
    # T21 grid
    num_fourier_default = 21  # Fourier wave number truncation
    nθ_default = 32  # Latitudinal grid size
    nd_default = 20  # Vertical slices

    start_time_default = 0
    end_time_default = 691200
    Δt_default = 1800

    function Barotropic_Main(num_fourier = num_fourier_default,
                             nθ = nθ_default,
                             nd = nd_default,
                             start_time = start_time_default,
                             end_time = end_time_default,
                             Δt = Δt_default)
        # the decay of a sinusoidal disturbance to a zonally symmetric flow
        # that resembles that found in the upper troposphere in Northern winter.
          name = "Barotropic"

          num_spherical = num_fourier + 1
          nλ = 2nθ  # Longitudinal points (?)

          radius = 6371.2e3
          omega = 7.292e-5

        # Initialize mesh
          mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
          θc, λc = mesh.θc,  mesh.λc
          cosθ, sinθ = mesh.cosθ, mesh.sinθ


        # Initialize atmo_data
          atmo_data = Atmo_Data(name, nλ, nθ, nd, false, false, false, false, sinθ, radius, omega)

        # Initialize integrator
          damping_order = 4
          damping_coef = 1.e-04
          robert_coef  = 0.04
          implicit_coef = 0.0

          init_step = true

          integrator = Filtered_Leapfrog(robert_coef,
                                 damping_order, damping_coef, mesh.laplacian_eig,
                                 implicit_coef,
                                 Δt, init_step, start_time, end_time)

        # Initialize data
          dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd, 0, 0)

          grid_u, grid_v = dyn_data.grid_u_c, dyn_data.grid_v_c

          grid_vor = dyn_data.grid_vor
          for i = 1:nλ
              grid_u[i, :, 1] .= 25 * cosθ - 30 * cosθ.^3 + 300 * sinθ.^2 .* cosθ.^6
          end
          grid_v .= 0.0
          spe_vor_c, spe_div_c = dyn_data.spe_vor_c, dyn_data.spe_div_c
          Vor_Div_From_Grid_UV!(mesh, grid_u, grid_v, spe_vor_c, spe_div_c)
          Trans_Spherical_To_Grid!(mesh, spe_vor_c,  dyn_data.grid_vor)
          Trans_Spherical_To_Grid!(mesh, spe_div_c,  dyn_data.grid_div)

        # ! adding a perturbation to the vorticity
          m, θ0, θw, A = 4.0, 45.0 * pi / 180, 15.0 * pi / 180.0, 8.0e-5
          for i = 1:nλ
              for j = 1:nθ
                  dyn_data.grid_vor[i,j, 1] += A / 2.0 * cosθ[j] * exp(-((θc[j] - θ0) / θw)^2) * cos(m * λc[i])
              end
          end
          Trans_Grid_To_Spherical!(mesh, dyn_data.grid_vor, spe_vor_c)

          NT = Int64(end_time / Δt)
          temporal_grid_dims = (nλ, nθ, nd, NT)

          # Initialize a var to store U, V for each time step
          temporal_grid_u = zeros(Float64, temporal_grid_dims)
          temporal_grid_v = zeros(Float64, temporal_grid_dims)

          time = start_time
          Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
          Update_Init_Step!(integrator)
          temporal_grid_u[:,:,:,1] = dyn_data.grid_u_c
          temporal_grid_v[:,:,:,1] = dyn_data.grid_v_c
          time += Δt
          for i = 2:NT
              Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
              time += Δt
              @info time
              # Store U, V each iter. NOTE: Not needed here, barotropic only solves at nd=1
              temporal_grid_u[:,:,:,i] = dyn_data.grid_u_c
              temporal_grid_v[:,:,:,i] = dyn_data.grid_v_c

          end
          return temporal_grid_u, temporal_grid_v
          Lat_Lon_Pcolormesh(mesh, grid_u,  1, "./data/data_plots/Barotropic_vel_u.png")
          Lat_Lon_Pcolormesh(mesh, grid_vor, 1, "./data/data_plots/Barotropic_vor.png")
    end

    function train_test(train_len = 192,
                        predict_len = 96,
                        shift = 96,
                        num_fourier = num_fourier_default,
                        nθ = nθ_default,
                        nd = nd_default,
                        start_time = start_time_default,
                        end_time = end_time_default,
                        Δt = Δt_default)
        # Run the barotropic model, store u & v data
        temporal_grid_u, temporal_grid_v = Barotropic_Main(num_fourier, nθ, nd, start_time, end_time, Δt)

        # Grab first vertical component (others aren't solved), flatten the remaining spatial components
        # Leaves us a 2D array of spatial solutions & time step
        data_u = reshape(temporal_grid_u[:,:,1,:], (:,size(temporal_grid_u)[4]))
        data_v = reshape(temporal_grid_v[:,:,1,:], (:,size(temporal_grid_v)[4]))
        train_u = data_u[:, shift:shift+train_len-1]
        train_v = data_v[:, shift:shift+train_len-1]
        test_u = data_u[:, shift+train_len:shift+train_len+predict_len-1]
        test_v = data_v[:, shift+train_len:shift+train_len+predict_len-1]

        return train_u, test_u, train_v, test_v

    end
end

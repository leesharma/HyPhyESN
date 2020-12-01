using JGCM

function hybridM_barotropic!(current_state, mesh, atmo_data, dyn_data):
    #-- Used in CHyPP
    #-- Takes input <current_state> containing ESN predicted u, v (flattened array)
    #-- Takes input mesh, dyn_data with additional state info
    #-- Returns the t+1 state predicted by the solver.

    # Initialize integrator. Assumes default values used to generate.
    damping_order = 4
    damping_coef = 1.15741e-4
    robert_coef  = 0.04
    implicit_coef = 0.0

    day_to_sec = 86400
    start_time = 0
    end_time = 1800  # Made this a single step, not sure about this
    Δt = 1800
    init_step = false

    integrator = Filtered_Leapfrog(robert_coef, damping_order, damping_coef,
                                    mesh.laplacian_eig, implicit_coef, Δt, init_step,
                                    start_time, end_time)

    # Load current state as grid_u_c and grid_v_c
    nθ = mesh.nθ
    new_u = current_state[1:Int64(size(current_state)[1]/2)] # TODO: check dims of input
    grid_u = reshape(new_u, (2*nθ, nθ, :))
    new_v = current_state[Int64(size(current_state)[1]/2+1):end]
    grid_v = reshape(new_v, (2*nθ, nθ, :))

    # Open variables to update with grid_u_c and grid_v_c
    dyn_data.grid_u_c = grid_u
    dyn_data.grid_v_c = grid_v
    spe_vor_c, spe_zeros = dyn_data.spe_vor_c, dyn_data.spe_zeros
    grid_vor = dyn_data.grid_vor
    grid_absvor = dyn_data.grid_absvor
    grid_δv = dyn_data.grid_δv
    grid_δu = dyn_data.grid_δu
    spe_δvor = dyn_data.spe_δvor
    spe_δdiv = dyn_data.spe_δdiv

    # Calculate spectral vorticity and divergence from grid_u/grid_v
    Vor_Div_From_Grid_UV!(mesh, grid_u, grid_v, spe_vor_c, spe_zeros)
    # Calculate grid vorticity from spectral
    Trans_Spherical_To_Grid!(mesh, spe_vor_c, grid_vor)
    # Calculate grid absvor from vor
    Compute_Abs_Vor!(grid_vor, atmo_data.coriolis, grid_absvor)
    # Calculate grid δu & δv
    grid_δu .= grid_absvor .* grid_v
    grid_δv .= -grid_absvor .* grid_u
    # Calculate spectral δvor & δdiv
    Vor_Div_From_Grid_UV!(mesh, grid_δu, grid_δv, spe_δvor, spe_δdiv)

    # Iterate the solver and get new predictions
    Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)

    # Return new predicted state by flattening & concatenating u & v
    next_state = cat(vec(dyn_data.grid_u_c), vec(dyn_data.grid_v_c), dims = 1)

    return next_state, mesh, atmo_data, dyn_data

end

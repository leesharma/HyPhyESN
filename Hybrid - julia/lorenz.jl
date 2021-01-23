# using Pkg; Pkg.activate("."); Pkg.instantiate()
# Pkg.add("ParameterizedFunctions")
# Pkg.add("OrdinaryDiffEq")

module LorenzData
  export train_test

  using ParameterizedFunctions, OrdinaryDiffEq

  # Default initial conditions
  u0_default = [1.0,0.0,0.0]
  tspan_default = (0.0,1000.0)
  p_default = [10.0,28.0,8/3]   # canonical chaotic regime

  function lorenz(du,u,p,t)
    # for readability
    sigma,rho,beta = p
    x,y,z = u

    du[1] = sigma*(y-x)
    du[2] = x*(rho-z) - y
    du[3] = x*y - beta*z
  end

  function lorenz_solution(u0=u0_default, tspan=tspan_default; p=p_default, dt=0.02, solver=ABM54())
    prob = ODEProblem(lorenz, u0, tspan, p)
    sol = solve(prob, solver, dt=dt)
    v = sol.u
    data = Matrix(hcat(v...))
  end

  function reduced_lorenz_step(u0=u0_default, ϵᵦ=0.05; p_base=p_default, dt=0.02, solver=ABM54())
    p_rom = p_base .* [1., 1., 1+ϵᵦ]
    reduced_lorenz_step_single(u) = lorenz_solution(u, (0.0, dt), p=p_rom, dt=dt, solver=solver)[:,end]

    hcat(map(reduced_lorenz_step_single, eachcol(u0))...)
  end

  # public interface

  function train_test(; train_len=5000, predict_len=1250, shift=300,      # split config
                        u0=u0_default, tspan=tspan_default, p=p_default,  # lorenz config
                        dt=0.02, solver=ABM54())                          # solver config
    data = lorenz_solution(u0, tspan, p=p, dt=dt, solver=solver)
    train = data[:, shift:shift+train_len-1]
    test = data[:, shift+train_len:shift+train_len+predict_len-1]
    (train, test)
  end
end

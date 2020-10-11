module LorenzData
  export train_test

  using ParameterizedFunctions, OrdinaryDiffEq

  # Default initial conditions
  u0_default = [1.0,0.0,0.0]
  tspan_default = (0.0,1000.0)
  p_default = [10.0,28.0,8/3]

  function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
  end

  function lorenz_solution(u0=u0_default, tspan=tspan_default; p=p_default, dt=0.02, solver=ABM54())
    prob = ODEProblem(lorenz, u0, tspan, p)
    sol = solve(prob, solver, dt=dt)
    v = sol.u
    data = Matrix(hcat(v...))
  end

  function train_test(; train_len=5000, predict_len=1250, shift=300,      # split config
                        u0=u0_default, tspan=tspan_default, p=p_default,  # lorenz config
                        dt=0.02, solver=ABM54())                          # solver config
    data = lorenz_solution(u0, tspan, p=p, dt=dt, solver=solver)
    train = data[:, shift:shift+train_len-1]
    test = data[:, shift+train_len:shift+train_len+predict_len-1]
    (train, test)
  end
end

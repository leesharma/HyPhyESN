module TrainHybrid
  """
  ESN training functions using the hybrid ridge solver
  """

  export train_hybrid

  using MLJLinearModels
  using ReservoirComputing: HESNtrain, Ridge


  function train_hybrid(hesn; beta=0.0, solver=MLJLinearModels.Analytical())
    ridge = Ridge(beta, solver)
    W_out = HESNtrain(ridge, hesn)
  end

end # module TrainHybrid

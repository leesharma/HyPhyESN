module BaseESN
  export esn, train, predict

  using ReservoirComputing

  # default model parameters
  approx_res_size = 300   # size of the reservoir
  radius = 1.2            # desired spectral radius
  activation = tanh       # neuron activation function
  degree = 6              # degree of connectivity of the reservoir
  sigma = 0.1             # input weight scaling
  beta = 0                # ridge
  alpha = 1.0             # leaky coefficient
  nla_type = NLAT2()      # non linear algorithm for the states
  extended_states = false # if true extends the states with the input

  # public interface

  function esn(data; approx_res_size = approx_res_size, beta = beta)
    """Create echo state network"""
    ESN(approx_res_size,
      data,
      degree,
      radius,
      activation = activation,
      sigma = sigma,
      alpha = alpha,
      nla_type = nla_type,
      extended_states = extended_states)
  end

  function train(esn)
    W_out = ESNtrain(esn, beta)
  end

  function predict(esn, predict_len, W_out)
    output = ESNpredict(esn, predict_len, W_out)
  end
end

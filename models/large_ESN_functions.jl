module LargeESN

    export large_ESN, large_ESNpredict

    using ReservoirComputing
    using Arpack: eigs
    using SparseArrays: sprand, sparse
    using LinearAlgebra: mul!

    function large_ESN(approx_res_size::Int,
            train_data::AbstractArray{T},
            degree::Int,
            radius::T;
            activation::Any = tanh,
            sigma::T = 0.1,
            alpha::T = 1.0,
            nla_type = NLADefault(),
            extended_states::Bool = false) where T<:AbstractFloat

        in_size = size(train_data, 1)
        out_size = size(train_data, 1)
        res_size = Int(floor(approx_res_size/in_size)*in_size)
        W = init_large_reservoir_givendeg(res_size, radius, degree)
        W_in = init_input_layer(res_size, in_size, sigma)
        states = large_states_matrix(W, W_in, train_data, alpha, activation, extended_states)

        return ESN{T, typeof(train_data),
            typeof(res_size),
            typeof(extended_states),
            typeof(activation),
            typeof(nla_type)}(res_size, in_size, out_size, train_data,
        alpha, nla_type, activation, W, W_in, states, extended_states)
    end

    #given degree of connections between neurons
    """
        init_large_reservoir_givendeg(res_size::Int, radius::Float64, degree::Int)
    Return a reservoir matrix scaled by the radius value and with a given degree of connection.
    Modifies the version used in ReservoirComputing to use Arpack eigs(), optimized
    for calculating eigenvalues of very large matrices
    """
    function init_large_reservoir_givendeg(res_size::Int,
                                           radius::Float64,
                                           degree::Int)

       sparsity = degree/res_size
       W = Matrix(sprand(Float64, res_size, res_size, sparsity))
       W = 2.0 .*(W.-0.5)
       replace!(W, -1.0=>0.0)
       W = sparse(W)
       # Taking the max of 3 converges better'
       rho_w = maximum(abs.(eigs(W, nev=3, which=:LM, maxiter = 100000, ritzvec=false)[1]))
       # Convert back to dense matrix for future operations
       W = Matrix(W)
       W .*= radius/rho_w
       return W
    end

    """
    ESNpredict(esn::AbstractLeakyESN, predict_len::Int, W_out::AbstractArray{Float64})
    Return the prediction for a given length of the constructed ESN struct.
    Optimized for large datasets.
    """
    function large_ESNpredict(esn,
                              predict_len::Int,
                              W_out::AbstractArray{Float64})

        output = zeros(Float64, esn.in_size, predict_len)
        x = esn.states[:, end]
        W = sparse(esn.W)

        if esn.extended_states == false
            for i=1:predict_len
                x_new = nla(esn.nla_type, x)
                out = (W_out*x_new)
                output[:, i] = out
                x = leaky_fixed_rnn(esn.activation, esn.alpha, W, esn.W_in, x, out)
            end
        elseif esn.extended_states == true
            for i=1:predict_len
                x_new = nla(esn.nla_type, x)
                out = (W_out*x_new)
                output[:, i] = out
                x = vcat(leaky_fixed_rnn(esn.activation, esn.alpha, W, esn.W_in, x[1:esn.res_size], out), out)
            end
        end
        return output
    end

    function large_states_matrix(W::AbstractArray{Float64},
        W_in::AbstractArray{Float64},
        train_data::AbstractArray{Float64},
        alpha::Float64,
        activation::Function,
        extended_states::Bool)

        train_len = size(train_data, 2)
        res_size = size(W, 1)
        in_size = size(train_data, 1)

        # Transform W into a sparse matrix for faster calculations
        W = sparse(W)

        # Initialize the states matrix
        states = zeros(Float64, res_size, train_len)

        # Multiply W_in*train_data in advance of the for loop
        W_in_x = Matrix{Float64}(undef, res_size, train_len)
        mul!(W_in_x, W_in, train_data)

        # Initialize an empty matrix to hold the W*states calculation
        y = Matrix{Float64}(undef, res_size, 1)

        for i=1:train_len-1
            states[:, i+1] = large_leaky_fixed_rnn(activation, alpha, W, W_in_x[:,i], states[:, i], y)
        end

        if extended_states == true
            ext_states = vcat(states, hcat(zeros(Float64, in_size), train_data[:,1:end-1]))
            return ext_states
        else
            return states
        end
    end

    function leaky_fixed_rnn(activation, alpha, W, W_in, x, y)
        return (1-alpha).*x + alpha*activation.((W*x)+(W_in*y))
    end

    function large_leaky_fixed_rnn(activation, alpha, W, W_in_x, states, y)
        mul!(y, W, states)
        return (1-alpha).*states + alpha*activation.(y+W_in_x)
    end

    """
        init_input_layer(res_size::Int, in_size::Int, sigma::Float64)
    Return a weighted input layer matrix, with random non-zero elements drawn from \$ [-\\text{sigma}, \\text{sigma}] \$, as described in [1].
    [1] Lu, Zhixin, et al. "Reservoir observers: Model-free inference of unmeasured variables in chaotic systems." Chaos: An Interdisciplinary Journal of Nonlinear Science 27.4 (2017): 041102.
    """
    function init_input_layer(res_size::Int,
            in_size::Int,
            sigma::Float64)

        W_in = zeros(Float64, res_size, in_size)
        q = floor(Int, res_size/in_size) #need to fix the reservoir input size. Check the constructor
        for i=1:in_size
            W_in[(i-1)*q+1 : (i)*q, i] = (2*sigma).*(rand(Float64, 1, q).-0.5)
        end
        return W_in

    end

end

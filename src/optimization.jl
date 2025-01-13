"""
    dispatch_parameters(circ, problem::Problem, beta_and_gamma)

Returns the circuit with the all parameters in the proper places.    

### Notes
- The number of driver parameters is the number of parameters in the circuit divided by the number of layers, minus the number of problem parameters.
- The macro `ChainRulesCore.@ignore_derivatives` is necessary because `Zygote` does not support automatic differentiation through mutating code.
"""
function dispatch_parameters(circ, problem::Problem, beta_and_gamma)
    @unpack_Problem problem

    #@assert nparameters(circ) == length(beta_and_gamma)

    #num_driver_parameters = (nparameters(circ) ÷ num_layers) - (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)

    #concat_params = l -> vcat(beta_and_gamma[l + num_layers] .* ChainRulesCore.@ignore_derivatives(problem_parameters(local_fields, couplings)),
    #                          beta_and_gamma[l] .* 2. .* ones(num_driver_parameters))
       
    
    #all_params = map(concat_params, 1:num_layers)
    
    all_params = beta_and_gamma
    
    circ = dispatch(circ, reduce(vcat, all_params))
    circ
end



"""
    problem_hamiltonian(problem::Problem)

Returns the problem Hamiltonian corresponding to `problem`.
"""
function problem_hamiltonian(problem::Problem)
    H =  sum([problem.local_fields[i] * put(i => Z)(problem.num_qubits) for i in 1:problem.num_qubits])
    H += sum([problem.couplings[i, j] * put((i, j) => kron(Z, Z))(problem.num_qubits) for j in 1:problem.num_qubits for i in 1:j-1])
    H
end


"""
Computes the cost function for VQE, which is the expectation value of the problem Hamiltonian.

### Input:
- `problem`: The problem instance, including the Hamiltonian.
- `params`: A vector of variational parameters, one for each gate in the circuit.

### Output:
- The expectation value of the problem Hamiltonian (a real number).
"""
function cost_function(problem::Problem, beta_and_gamma::Vector{Float64})::Real
    circ = ChainRulesCore.@ignore_derivatives(circuit(problem))
    circ = dispatch_parameters(circ, problem, beta_and_gamma)
    reg = apply(uniform_state(nqubits(circ)), circ)
    expect(ChainRulesCore.@ignore_derivatives(problem_hamiltonian(problem)), reg) |> real
end


"""
Optimizes the parameters for VQE using gradient ascent.

### Input:
- `problem`: The problem instance, including the Hamiltonian.
- `params`: Initial parameters for the circuit.
- `niter`: Number of iterations for the optimization loop.
- `learning_rate`: Step size for parameter updates.

### Output:
- `cost`: Final value of the cost function.
- `params`: Optimized parameters.
- `probabilities`: Measurement probabilities from the final circuit.

### Notes:
- The optimization uses gradient ascent to minimize the cost function.
"""
function optimize_parameters(problem::Problem, ini_params::Vector{Float64}; niter::Int=128, learning_rate::Float64=0.05)

    f = x -> cost_function(problem, x)

    cost_history = []
    
    cost = f(ini_params)
    params = ini_params
    #println("Initial Cost: ",cost)
    for n in 1:niter
        params = params .+ learning_rate .* gradient(f, params)[1]
        cost = f(params)

        push!(cost_history, copy(cost))
    end

    # Construct the final circuit with optimized parameters
    circ = circuit(problem)
    circ = dispatch_parameters(circ, problem, params)

    # Compute measurement probabilities
    probabilities = uniform_state(nqubits(circ)) |> circ |> Yao.probs

    cost, params, probabilities, cost_history
end

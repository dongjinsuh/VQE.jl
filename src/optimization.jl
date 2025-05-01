"""
    problem_parameters(local_fields::Vector{Real}, couplings::Matrix{Real})::Vector{Real}

Returns the problem parameters in the proper order, such that they can be `dispatch`ed to the circuit directly.
"""
function problem_parameters(local_fields::Vector{Real}, couplings::Matrix{Real})::Vector{Real}
    num_qubits = size(local_fields)[1]
    vcat(2 .* local_fields, [2 * couplings[i, j] for j in 1:num_qubits for i in 1:j-1])
end



"""
    dispatch_parameters(circ, problem::Problem, parameters)

Returns the circuit with the all parameters in the proper places.    

"""
function dispatch_parameters(circ, problem::Problem, parameters)
    @unpack_Problem problem

    #@assert nparameters(circ) == length(parameters)

    num_driver_params = (nparameters(circ) ÷ num_layers) - (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)
    num_problem_params = (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)

    num_total_params = num_driver_params + num_problem_params

    concat_params = l -> vcat(parameters[(1+(l-1)*num_total_params+num_driver_params):((l-1)*num_total_params+num_total_params)] .* ChainRulesCore.@ignore_derivatives(problem_parameters(local_fields, couplings)),
                              parameters[(1+(l-1)*num_total_params):((l-1)*num_total_params+num_driver_params)] .* ones(num_driver_params))
    
    all_params = map(concat_params, 1:num_layers)
    #println(all_params)
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
function cost_function(problem::Problem, params::Vector{Float64})::Real
    circ = ChainRulesCore.@ignore_derivatives(circuit(problem))
    circ = dispatch_parameters(circ, problem, params)
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
- The optimization uses gradient ascent to maximize the cost function.
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


####################





function problem_parameters_single(local_fields::Vector{Real}, couplings::Matrix{Real})::Vector{Real}
    num_qubits = size(local_fields)[1]
    #vcat(2 .* local_fields)
    vcat(local_fields)
end

function problem_parameters_coupling(local_fields::Vector{Real}, couplings::Matrix{Real})::Vector{Real}
    num_qubits = size(local_fields)[1]
    #vcat([2 * couplings[i, j] for j in 1:num_qubits for i in 1:j-1])
    vcat([couplings[i, j] for j in 1:num_qubits for i in 1:j-1])

end



function dispatch_parameters_mix(circ, problem::Problem, parameters)
    @unpack_Problem problem

    num_driver_params = (nparameters(circ) ÷ num_layers) - (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)

    num_z_params = num_qubits

    num_zz_params = 1

    num_problem_params = num_z_params + num_zz_params

    num_total_params = num_driver_params + num_problem_params

    concat_params = l -> vcat(parameters[(1+(l-1)*num_total_params+num_driver_params):((l-1)*num_total_params+num_driver_params+num_z_params)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)),
                              parameters[(1+(l-1)*num_total_params+num_driver_params+num_z_params)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_coupling(local_fields, couplings)),
                              parameters[(1+(l-1)*num_total_params):((l-1)*num_total_params+num_driver_params)] .* ones(num_driver_params))

    all_params = map(concat_params, 1:num_layers)
    #println(all_params)
    circ = dispatch(circ, reduce(vcat, all_params))
    circ
end



function cost_function_mix(problem::Problem, params::Vector{Float64})::Real
    circ = ChainRulesCore.@ignore_derivatives(circuit(problem))
    circ = dispatch_parameters_mix(circ, problem, params)
    reg = apply(uniform_state(nqubits(circ)), circ)
    expect(ChainRulesCore.@ignore_derivatives(problem_hamiltonian(problem)), reg) |> real
end


#number of parameters = 2*N + 1 
function optimize_parameters_mix(problem::Problem, ini_params::Vector{Float64}; niter::Int=128, learning_rate::Float64=0.05)

    f = x -> cost_function_mix(problem, x)

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
    circ = dispatch_parameters_mix(circ, problem, params)
    # Compute measurement probabilities
    probabilities = uniform_state(nqubits(circ)) |> circ |> Yao.probs

    cost, params, probabilities, cost_history
end


######################

# Two params optimization for higher N


function dispatch_parameters_two_params(circ, problem::Problem, fix_parameters, params)
    @unpack_Problem problem

    #num_driver_params = (nparameters(circ) ÷ num_layers) - (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)

    num_driver = Int(num_qubits)
    num_z = Int(num_qubits)
    #println(num_driver)
    #println(num_z)
    num_zz = 1

    num_problem = num_z + num_zz
    
    # for the fix parameters: -4 because the for rach layers two x- and two z-term parameters are optimized and therefore not fixed
    num_total_params = num_driver + num_problem - 4
    #println(num_total_params)

    
    concat_params = l -> vcat(fix_parameters[(l-1)*num_total_params+num_driver-1:(l-1)*num_total_params+num_driver+num_z-4] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[1:num_z-2]),
                              params[1+(l-1)*4] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z-1]),
                              params[2+(l-1)*4] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z]),
 
                              fix_parameters[((l-1)*num_total_params+num_driver+num_z-3)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_coupling(local_fields, couplings)),

                              fix_parameters[1+(l-1)*num_total_params:(l-1)*num_total_params+num_driver-2] .* ones(num_driver-2),
                              params[3+(l-1)*4] .* ones(1),
                              params[4+(l-1)*4] .* ones(1))

    #println("a", ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z-1])) 
    #println("b", ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z])) 
    #println("c", ones(1)) 
    #println("d", ones(1))

    all_params = map(concat_params, 1:num_layers)

    #println(all_params)
    circ = dispatch(circ, reduce(vcat, all_params))
    circ
end



#::Vector{Float64}
function cost_function_two(problem::Problem, fix_parameters, params)::Real
    circ = ChainRulesCore.@ignore_derivatives(circuit(problem))
    circ = dispatch_parameters_two_params(circ, problem, fix_parameters, params)
    reg = apply(uniform_state(nqubits(circ)), circ)
    expect(ChainRulesCore.@ignore_derivatives(problem_hamiltonian(problem)), reg) |> real
end



function optimize_parameters_mix_two(problem::Problem, ini_params::Vector{Float64}; niter::Int=128, learning_rate::Float64=0.05)
    @unpack_Problem problem

    # number of parameters per layer
    num_params = Int(length(ini_params)/num_layers)

    params = []
    indices_to_remove = Int[]

    for i in 0:(num_layers-1)  # Julia indexing starts at 1, so adjust accordingly
        base_index = i * num_params  # Start index of the current layer

        push!(params, ini_params[base_index + Int((num_params-1)/2 - 1)])  # sec last element in x term
        push!(params, ini_params[base_index + Int((num_params-1)/2)])  # last element in x term
        push!(params, ini_params[base_index + Int(num_params-2)]) # sec element in z term
        push!(params, ini_params[base_index + Int(num_params-1)]) # last element in z term
        
        
        push!(indices_to_remove, base_index + Int((num_params-1)/2 - 1))
        push!(indices_to_remove, base_index + Int((num_params-1)/2))
        push!(indices_to_remove, base_index + Int(num_params-2))
        push!(indices_to_remove, base_index + Int(num_params-1))
    end

    sort!(indices_to_remove)

    fix_parameters = deleteat!(copy(ini_params), indices_to_remove)
    
    #println(fix_parameters)
    #println(params)

    f = x -> cost_function_two(problem, fix_parameters, x)

    cost_history = []
    
    cost = f(params)
    #params = ini_params
    #println("Initial Cost: ",cost)
    for n in 1:niter
        params = params .+ learning_rate .* gradient(f, params)[1]
        #println(gradient(f, params)[1])
        #println(params)
        cost = f(params)

        push!(cost_history, copy(cost))
    end

    # Construct the final circuit with optimized parameters
    circ = circuit(problem)
    circ = dispatch_parameters_two_params(circ, problem, fix_parameters, params)
    
    # Compute measurement probabilities
    probabilities = uniform_state(nqubits(circ)) |> circ |> Yao.probs

    # insert the optimized parameters back to the fix parameters
    for (idx, val) in zip(indices_to_remove, params)
        insert!(fix_parameters, idx, val)
    end

    cost, fix_parameters, probabilities, cost_history#, circ
end











########################### Reduced parameter numbers: N + 3 * p ####################









function dispatch_parameters_mix_half(circ, problem::Problem, parameters)
    @unpack_Problem problem

    #num_driver_params = (nparameters(circ) ÷ num_layers) - (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)

    num_driver_params = Int(num_qubits / 2) + 1
    num_z_params = Int(num_qubits / 2) + 1
    num_zz_params = 1

    num_problem_params = num_z_params + num_zz_params
    
    num_total_params = num_driver_params + num_problem_params

    concat_params = l -> vcat(parameters[(1+(l-1)*num_total_params+num_driver_params):((l-1)*num_total_params+num_driver_params+num_z_params-1)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[1:num_z_params-1]),
                              parameters[(1+(l-1)*num_total_params+num_driver_params+num_z_params)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z_params:num_qubits]),
                              parameters[(1+(l-1)*num_total_params+num_driver_params+num_z_params)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_coupling(local_fields, couplings)),
                              parameters[(1+(l-1)*num_total_params):((l-1)*num_total_params+num_driver_params-1)] .* ones(num_driver_params-1),
                              parameters[(1+(l-1)*num_total_params+num_driver_params)] .* ones(num_driver_params-1))
    

    all_params = map(concat_params, 1:num_layers)

    circ = dispatch(circ, reduce(vcat, all_params))
    circ
end


function cost_function_half(problem::Problem, parameters::Vector{Float64})::Real
    circ = ChainRulesCore.@ignore_derivatives(circuit(problem))
    circ = dispatch_parameters_mix_half(circ, problem, parameters)
    reg = apply(uniform_state(nqubits(circ)), circ)
    expect(ChainRulesCore.@ignore_derivatives(problem_hamiltonian(problem)), reg) |> real
end




function optimize_parameters_half(problem::Problem, ini_params::Vector{Float64}; niter::Int=128, learning_rate::Float64=0.05)

    f = x -> cost_function_half(problem, x)

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
    circ = dispatch_parameters_mix_half(circ, problem, params)

    # Compute measurement probabilities
    probabilities = uniform_state(nqubits(circ)) |> circ |> Yao.probs

    cost, params, probabilities, cost_history#, circ
end




#######################


# Two parameters Optimization for higher N 

function dispatch_parameters_two_params_v2(circ, problem::Problem, fix_parameters, params)
    @unpack_Problem problem

    #num_driver_params = (nparameters(circ) ÷ num_layers) - (num_qubits + num_qubits * (num_qubits - 1) ÷ 2)

    num_driver = Int(num_qubits / 2)
    num_z = Int(num_qubits / 2)
    #println(num_driver)
    #println(num_z)
    num_zz = 1

    num_problem = num_z + num_zz
    
    num_total_params = num_driver + num_problem - 2
    #println(num_total_params)

    
    concat_params = l -> vcat(fix_parameters[(l-1)*num_total_params+num_driver:(l-1)*num_total_params+num_driver+num_z-2] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[1:num_z-1]),
                              params[1+(l-1)*4] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z]),
                              params[2+(l-1)*4] .* ChainRulesCore.@ignore_derivatives(problem_parameters_single(local_fields, couplings)[num_z+1:num_qubits]),
 
                              fix_parameters[((l-1)*num_total_params+num_driver+num_z-1)] .* ChainRulesCore.@ignore_derivatives(problem_parameters_coupling(local_fields, couplings)),

                              fix_parameters[1+(l-1)*num_total_params:(l-1)*num_total_params+num_driver-1] .* ones(num_driver-1),
                              params[3+(l-1)*4] .* ones(1),
                              params[4+(l-1)*4] .* ones(num_driver))
    

    all_params = map(concat_params, 1:num_layers)
    #println(all_params)
    circ = dispatch(circ, reduce(vcat, all_params))
    circ
end



#::Vector{Float64}
function cost_function_two_v2(problem::Problem, fix_parameters, params)::Real
    circ = ChainRulesCore.@ignore_derivatives(circuit(problem))
    circ = dispatch_parameters_two_params_v2(circ, problem, fix_parameters, params)
    reg = apply(uniform_state(nqubits(circ)), circ)
    expect(ChainRulesCore.@ignore_derivatives(problem_hamiltonian(problem)), reg) |> real
end



function optimize_parameters_two_v2(problem::Problem, ini_params::Vector{Float64}; niter::Int=128, learning_rate::Float64=0.05)
    @unpack_Problem problem

    # number of parameters per layer
    num_params = Int(length(ini_params)/num_layers)

    params = []
    indices_to_remove = Int[]

    for i in 0:(num_layers-1)  # Julia indexing starts at 1, so adjust accordingly
        base_index = i * num_params  # Start index of the current layer

        push!(params, ini_params[base_index + Int((num_params-1)/2 - 1)])  # 6th element in x term
        push!(params, ini_params[base_index + Int((num_params-1)/2)])  # 7th element in x term
        push!(params, ini_params[base_index + Int(num_params-2)]) # 6th element in z term
        push!(params, ini_params[base_index + Int(num_params-1)]) # 7th element in z term
        
        
        push!(indices_to_remove, base_index + Int((num_params-1)/2 - 1))
        push!(indices_to_remove, base_index + Int((num_params-1)/2))
        push!(indices_to_remove, base_index + Int(num_params-2))
        push!(indices_to_remove, base_index + Int(num_params-1))
    end

    sort!(indices_to_remove)

    fix_parameters = deleteat!(copy(ini_params), indices_to_remove)

    #println(fix_parameters)
    #println(params)

    f = x -> cost_function_two_v2(problem, fix_parameters, x)

    cost_history = []
    
    cost = f(params)
    #params = ini_params
    #println("Initial Cost: ",cost)
    for n in 1:niter
        params = params .+ learning_rate .* gradient(f, params)[1]
        cost = f(params)

        push!(cost_history, copy(cost))
    end

    # Construct the final circuit with optimized parameters
    circ = circuit(problem)
    circ = dispatch_parameters_two_params_v2(circ, problem, fix_parameters, params)
    
    # Compute measurement probabilities
    probabilities = uniform_state(nqubits(circ)) |> circ |> Yao.probs

    # insert the optimized parameters back to the fix parameters
    for (idx, val) in zip(indices_to_remove, params)
        insert!(fix_parameters, idx, val)
    end

    cost, fix_parameters, probabilities, cost_history#, circ
end
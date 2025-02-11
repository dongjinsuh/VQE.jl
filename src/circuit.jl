"""
    local_field_gates(num_qubits::Int, operator=Z)

Returns the single-qubit rotation gates corresponding to the local part of the problem (driver) Hamiltonian.

### Notes
- The gates are initialized with trivial parameters, which are then overwritten explicitly elsewhere.
"""
function local_field_gates(num_qubits::Int, operator=Z)
    [put(i => RotationGate(operator, 0.))(num_qubits) for i in 1:num_qubits] |> chain
end


"""
    coupling_gates(num_qubits::Int, operators=[Z, Z])

Returns the two-qubit rotation gates corresponding to the coupling term of the Hamiltonian. 

### Notes
- The gates are initialized with trivial parameters, which are then overwritten explicitly elsewhere.
"""
function coupling_gates(num_qubits::Int, operators=[Z, Z])
    [put((i, j) => RotationGate(kron(operators...), 0.))(num_qubits) for j in 1:num_qubits for i in 1:j-1] |> chain
end


"""
    coupling_gates(edges, num_qubits::Int, operators)

This function is used to construct two-qubit drivers such as ``\\hat X_i \\hat X_j``.

### Notes
- The gates are initialized with trivial parameters, which are then overwritten explicitly elsewhere.
"""
function coupling_gates(edges, num_qubits::Int, operators)
    [put(edge => RotationGate(kron(operators...), 0.))(num_qubits) for edge in edges] |> chain
end

"""
    objective_function(num_qubits::Int)    

Returns the gates corresponding to the entire problem Hamiltonian.    
"""
function objective_function(num_qubits::Int)
    [local_field_gates(num_qubits), coupling_gates(num_qubits)] |> chain
end


"""
    driver(num_qubits::Int, G::YaoBlocks.PauliGate)

Returns the gates for one driver layer constructed from a Pauli matrix, e.g. the standard driver has `G==X`.
"""
function driver(problem::Problem, G::YaoBlocks.PauliGate)
    @unpack_Problem problem
    local_field_gates(num_qubits, G)
end


"""
    driver(num_qubits::Int, G::Vector{<:YaoBlocks.PauliGate})

Returns the gates for one driver layer constructed from the Kronecker product of two Pauli matrices, 
e.g., `G==[X, X]` corresponds to drivers ``\\hat X_i \\hat X_j`` on all pairs of connected qubits.
"""
function driver(problem::Problem, G::Vector{<:YaoBlocks.PauliGate})
    @unpack_Problem problem
    @assert size(G)[1] == 2
    coupling_gates(edges, num_qubits, G)
end



"""
    layer(problem::Problem)

Returns the circuit corresponding to one QAOA layer.
"""
function layer(problem::Problem)
    [objective_function(problem.num_qubits), 
     driver(problem, problem.driver)] |> chain
end


"""
    circuit(problem::Problem)

Returns the `Yao` circuit for the given QAOA problem. 
"""    
function circuit(problem::Problem)
    [layer(problem) for _ in 1:problem.num_layers] |> chain
end
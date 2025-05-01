# script to generate the coupling matrices from Crosson's instances
using CSV, HDF5, Printf
using DelimitedFiles, Combinatorics

PATH = raw"/home/ubuntu/MAX2SAT/MAX2SATQuantumData/Mirkarimi_ComparingHardness/"

# dictionary for mapping clauses to Ising
# maps "type_i" to (local_field_1, local_field_2, coupling)
types_to_values = Dict("type_1" => (1, 1, 1), "type_2" => (1, -1, -1), "type_3" => (-1, 1, -1), "type_4" => (-1, -1, 1))

# helper function to map data connectives to the above type dict
function get_clause_type_from(pair_of_unary_logical_connectives)
    mapping = Dict((1, 1) => "type_1", (1, -1) => "type_2", (-1, 1) => "type_3", (-1, -1) => "type_4")
    mapping[pair_of_unary_logical_connectives]
end

# ================================================================================================================

# read overview fike
csv_reader = CSV.File(PATH * "instances_typical.csv")

# put hashes into dict according to number of qubits
nqubits_to_ID = Dict()

for nqubits in csv_reader[1][2]:csv_reader[end][2]
    nqubits_to_ID[nqubits] = []
end

for row in csv_reader
    push!(nqubits_to_ID[row[2]], row[1])
end

# ================================================================================================================

N = 16
num_clauses = 3N

for idx in 1:1000
    filename = PATH * "instances_typical/" * @sprintf("%s.m2s", nqubits_to_ID[N][idx])
    printstyled("instance ", filename, "\n", color=:blue)

    rawdata = readdlm(filename)

    not_negated_1 = rawdata[:, 1]
    not_negated_2 = rawdata[:, 3]
    unary_logical_connectives = zip(not_negated_1, not_negated_2) |> collect 
    clause_types = get_clause_type_from.(unary_logical_connectives)
 
    vars_1 = (rawdata[:, 2] .|> Int) .+ 1
    vars_2 = (rawdata[:, 4] .|> Int) .+ 1    
    clause_vars = zip(vars_1, vars_2) .|> collect

    clause_types = get_clause_type_from.(unary_logical_connectives)
    
    C = zip(clause_vars, clause_types)  |> collect

    h = zeros(N)
    J = zeros(N, N)
    for c in C
        h[c[1][1]] += -types_to_values[c[2]][1]
        h[c[1][2]] += -types_to_values[c[2]][2]
        J[c[1][1], c[1][2]] += -types_to_values[c[2]][3]
    end
    J_mat = J + transpose(J)


    PATH_w = raw"/home/ubuntu/MAX2SAT/MAX2SATQuantumData/Mirkarimi_data/HDF5"


    h5write(PATH_w * @sprintf("/max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses), "not_negated_vars_1", not_negated_1)
    h5write(PATH_w * @sprintf("/max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses), "not_negated_vars_2", not_negated_2)
    h5write(PATH_w * @sprintf("/max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses), "vars_1", vars_1)
    h5write(PATH_w * @sprintf("/max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses), "vars_2", vars_2)
    h5write(PATH_w * @sprintf("/max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses), "local_fields", h)
    h5write(PATH_w * @sprintf("/max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses), "coupling_matrix", J_mat)
end
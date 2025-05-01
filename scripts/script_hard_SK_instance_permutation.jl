using QAOA
using DifferentialEquations
using SpinFluctuations
using Arpack    
using PyPlot 
using HDF5 
using Printf
using LinearAlgebra, Random, Distributions
using Dates, Crayons
using Revise
using SparseArrays
using Parameters
using Interpolations
using DataFrames


function energies_and_bitstrings_qaoa(annealing_problem)
    L = annealing_problem.num_qubits
    h = annealing_problem.local_fields
    J = annealing_problem.couplings
    
    bit_string_df = DataFrame( bit_string = [], energy = Float64[]);
    
    bitstrings = [string(i, base=2, pad=L) |> reverse for i in 0:(2^L - 1)]
    bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
    spins = [1 .- 2s for s in bitvals]
    
    for spin in spins
        energy = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
        push!(bit_string_df,[ spin, energy])
    end
    
    return bit_string_df
end


# load the SK hard instance data

# number of spins
N = 18
#N = parse(Int, ARGS[1])

PATH = raw"/home/ubuntu/aqc_QAOA/SpinFluctuations.jl"
subdir = "small_gaps"
# subdir = "large_gaps"
folder_name = PATH * @sprintf("//data//N_%i//", N);

### change N in the pattern individually 
pattern = r"hard_random_SK_instance_N_18_seed_(\d+)\.h5"
###

instance_names = readdir(folder_name)
loop_var = 1
#loop_var = parse(Int, ARGS[2])
total_num_inst = 0

#N12
# training data +0 - +70 # +500 +543
# test data +1000 +1200

#N14
#0 - 200

#N16
#0 - 200

#N18
#0 - 109

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+109])

    global total_num_inst = k
    println("current instance count: ", k)

    seed = match(pattern, instance_name)[1]
    seed = parse(Int64, seed)
    #seed = 19
    println("seed: ",seed)
    
    #spin_idx = 2
    file_name = folder_name * @sprintf("hard_random_SK_instance_N_%i_seed_%i.h5", N , seed)

    #gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    #adiabatic_peak = h5read(file_name, "adiabatic_peak")
    mf_problem = Problem(0, local_fields, J_mat);

    """
    # Exact Energie spectrum
    nev = 50
    keep_EVs = 10
    time_step = 33

    exact_times = range(0, 1, time_step)
    eigeninfo = map(s -> (eigs(-hamiltonian(1 - s, s, mf_problem.local_fields, mf_problem.couplings), nev=nev, which=:LM, maxiter=10000)), exact_times)
    lambda_s = [vals[1] for vals in eigeninfo]
    lambda = sort(reduce(hcat, lambda_s), dims=1)

    # without fixing a spin
    #all_eigvecs = zeros(length(exact_times), 2^(N-1), keep_EVs)
    all_eigvecs = zeros(length(exact_times), 2^(N), keep_EVs)
    for k in 1:length(exact_times)
        sorting_perm = sortperm(lambda_s[k])
        all_eigvecs[k, :, :] .= eigeninfo[k][2][:, sorting_perm[1:keep_EVs]]
    end
    """

    # mean field evolution

    T_final = 32768.
    tol = 1e-6

    schedule(t) = t / T_final
    sol = evolve_mean_field(mf_problem.local_fields, mf_problem.couplings, T_final, schedule, rtol=1e2tol, atol=tol) 

    # get mean-field solution
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    mf_sol = solution(sol(T_final)) 

    writable_data = zeros(length(sol.u), size(sol.u[1])...)
    for i in 1:length(sol.u)
        writable_data[i, :, :] .= sol.u[i]
    end

    sol_t = sol.t
    sol_u = writable_data

    nx_vals = n_vals("x", sol_u)
    ny_vals = n_vals("y", sol_u)
    nz_vals = n_vals("z", sol_u);



    # sort the spins by their degree of frustration

    npts = 2048
    coarse_times = range(0, 1, npts + 1)

    # From area under magnetization
    #nxy_coarse = zeros(N-1)
    nxy_coarse = zeros(N)

    nx_coarse = n_coarse(nx_vals, sol_t, coarse_times)
    ny_coarse = n_coarse(ny_vals, sol_t, coarse_times)
    nz_coarse = n_coarse(nz_vals, sol_t, coarse_times);

    S_vals = [transpose(reduce(hcat, [nxy_coarse, nxy_coarse, nz_coarse[:, k]])) |> Matrix for k in 1:npts+1]
    magnetizations = reduce(hcat, map(S -> magnetization(S, mf_problem.local_fields, mf_problem.couplings), S_vals));  

    # Get most frustrated spins from area under magnetization
    areas = Dict()
    dts = [(x[2] - x[1]) / T_final for x in zip(coarse_times[1:end-1], coarse_times[2:end])]

    #for spin_idx in 1:N-1
    for spin_idx in 1:N  
        areas[spin_idx] = sum(dts .* magnetizations[spin_idx, 2:end]) |> abs
    end
    all_most_frustrated_spins = [k for (k, v) in sort(areas |> collect, by=x->x[2])]

    # From area under Edwards-Anderson order parameter
    nzs = reduce(hcat, [sol_u[k, 3, :] for k in 1:size(sol_u)[1]])

    # Get "most undecided spin" from area under z components
    EA_param = Dict()
    dts = [(x[2] - x[1]) / T_final for x in zip(sol_t[1:end-1], sol_t[2:end])]

    #for spin_idx in 1:N-1
    
    for spin_idx in 1:N 
        EA_param[spin_idx] = sum(dts .* nzs[spin_idx, 2:end] .^ 2) |> abs
    end
    all_most_undecided_spins = [k for (k, v) in sort(EA_param |> collect, by=x->x[2])]    
        
    top_idxs = [k for (k, v) in sort(areas |> collect, by = x -> x[2])]
    #top_idxs = [k for (k, v) in sort(EA_param |> collect, by = x -> x[2])]
    top_idx = top_idxs[1]

    regular_trajectories = filter!(x -> x != top_idx, collect(1:N));



    # permute the spin labels in the interaction matrix and local fields

    # create the permutation order from the degree of frustration
    #perm_order = all_most_frustrated_spins  # need to fix, all_most_frustrated_spins is not correct
    perm_order = top_idxs

    J_mat_permuted = zeros(size(J_mat))

    # permute the rows and columns of J_mat 
    for i in 1:length(perm_order)
        for j in 1:length(perm_order)
            J_mat_permuted[i,j] = J_mat[perm_order[i], perm_order[j]]
        end
    end

    local_fields_permuted = zeros(size(local_fields))

    for i in 1:length(perm_order)
        local_fields_permuted[i] = local_fields[perm_order[i]]
    end

    mf_problem_permuted = Problem(0, local_fields_permuted, J_mat_permuted)


    # calculate the exact ground energy of the transformed instance
    exact_solution_bitstring = energies_and_bitstrings_qaoa(mf_problem_permuted)

    sorted_df = sort(exact_solution_bitstring, :energy)
    lowest_energy_row = sorted_df[1, :]
    gs_energy = lowest_energy_row[2]
    println(gs_energy)
    println(perm_order)

    # Apply the transformations to the instances and load into file_name

    # open file

    PATH_w = raw"/home/ubuntu/aqc_QAOA/transformation_hard_SK_instances"
    #subdir = "small_gaps"

    folder_name_w = PATH_w * @sprintf("//permutation_data//N_%i//", N);
    #folder_name_w = PATH_w * @sprintf("//permutation_test_data//N_%i//", N);


    # Open the file in read-write mode and overwrite the dataset

    h5open(folder_name_w * @sprintf("transformed_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
        write(file, "couplings", J_mat_permuted)                    # Save seed
        write(file, "local_fields", local_fields_permuted)         # Save cost values
        write(file, "ground_state_energy", gs_energy)        # Save parameters
        #write(file, "time_step", time_step)
        write(file, "permutation_order", perm_order)
    end

end
println("============================================================ N = ", N, ", ", total_num_inst," instances, done! ============================================================")

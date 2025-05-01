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


# load the MAX2SAT hard instance data

N = 16
#N = parse(Int, ARGS[1])
num_clauses = 3N

PATH = raw"/home/ubuntu/MAX2SAT/MAX2SATQuantumData/Mirkarimi_data/HDF5/"


#subdir = "small_gaps"
# subdir = "large_gaps"

#folder_name = PATH * @sprintf("//transformation_hard_SK_instances//data//N_%i//", N);

instance_names = collect(308:310)

total_num_inst = 0

#loop_var = parse(Int, ARGS[3])


# N12 re-run with diff time steps:
# 4 9 18 20 22 24 25 30 32 36 38 42 44 53 56 57 60 63 77 79 82 86 92

# N16: 44, 111, 266, 271, 277 285, 303 307
#      218, 233, 244, 249, 262, 265, 295
#      ??? 220, 225, 214, 236, 240, 288

# N14: 248 288 338 518 564
#       ??? 405 519 529 554 573 611 615 616 618

for (k, idx) in enumerate(instance_names[:])
    
    #print(idx, "\t")
    global total_num_inst = k
    println("current instance count: ", idx)

    instance_str = @sprintf("max2sat_typical_instance_%04i_from_arxiv_2206_06876_N_%i_num_clauses_%i.h5", idx, N, num_clauses)
    local_fields = h5read(PATH * instance_str, "local_fields")
    J_mat = h5read(PATH * instance_str, "coupling_matrix")

    mf_problem = Problem(0, local_fields, J_mat);

    time_step = 31
    s_gap = 0.0
    ratio = s_gap / (1 - s_gap)
    J_mat_ratio = J_mat*ratio
    local_fields_ratio = local_fields*ratio
    global mf_problem_ratio = Problem(0, local_fields_ratio, J_mat_ratio)


    # set the loop range reasoned 
    #for i in 31:31

    # Exact Energie spectrum
    #println(i)
    nev = 100

    keep_EVs = 10

    #time_step = i
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

    # calculate adiabatic ratio
    gap = lambda[2, :] .- lambda[1, :];
    mingap = minimum(gap)
    gap_idx = findfirst(x -> x == mingap, gap)
    s_gap = exact_times[gap_idx]

    

    # Apply a transformation to the instances such that s_* = 0.5
    # transform the J and h by ratio*J and ratio*h

    # ratio: s_* / (1 - s_*)
    ratio = s_gap / (1 - s_gap)
    #println(ratio)

    J_mat_ratio = J_mat*ratio
    local_fields_ratio = local_fields*ratio
    global mf_problem_ratio = Problem(0, local_fields_ratio, J_mat_ratio)

    ### check if s_gap is truely at 0.5
    eigeninfo = map(s -> (eigs(-hamiltonian(1 - s, s, mf_problem_ratio.local_fields, mf_problem_ratio.couplings), nev=nev, which=:LM, maxiter=10000)), exact_times)
    lambda_s = [vals[1] for vals in eigeninfo]
    lambda = sort(reduce(hcat, lambda_s), dims=1)
    
    gap = lambda[2, :] .- lambda[1, :];
    mingap = minimum(gap)
    gap_idx = findfirst(x -> x == mingap, gap)
    s_gap = exact_times[gap_idx]    
    

    # if s_gap is  0.5 exit the loop and continue to next step
    println("s_* = ", s_gap)
    """
    if s_gap == 0.5 
        println("s_* = ", s_gap)
        println("time_step: ", time_step)
        #println("current instance count: ", k)
        global total_num_inst = k # if s_gap never reaches 0.5 the number of instance will not be counted
        break
    end
    """

    #end
    
    
    # mean field evolution

    T_final = 32768.
    tol = 1e-6

    schedule(t) = t / T_final
    sol = evolve_mean_field(mf_problem_ratio.local_fields, mf_problem_ratio.couplings, T_final, schedule, rtol=1e2tol, atol=tol) 

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
    magnetizations = reduce(hcat, map(S -> magnetization(S, mf_problem_ratio.local_fields, mf_problem_ratio.couplings), S_vals));  

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

    J_mat_permuted = zeros(size(J_mat_ratio))

    # permute the rows and columns of J_mat 
    for i in 1:length(perm_order)
        for j in 1:length(perm_order)
            J_mat_permuted[i,j] = J_mat_ratio[perm_order[i], perm_order[j]]
        end
    end

    local_fields_permuted = zeros(size(local_fields_ratio))

    for i in 1:length(perm_order)
        local_fields_permuted[i] = local_fields_ratio[perm_order[i]]
    end

    mf_problem_permuted = Problem(0, local_fields_permuted, J_mat_permuted)


    # calculate the exact ground energy of the transformed instance
    exact_solution_bitstring = energies_and_bitstrings_qaoa(mf_problem_permuted)

    sorted_df = sort(exact_solution_bitstring, :energy)
    lowest_energy_row = sorted_df[1, :]
    gs_energy = lowest_energy_row[2]
    println(gs_energy)


    # Apply the transformations to the instances and load into file_name

    # open file

    PATH_w = raw"/home/ubuntu/MAX2SAT/MAX2SAT_transformation_instances"
    #subdir = "small_gaps"

    folder_name_w = PATH_w * @sprintf("//trans_test//N_%i//", N); # fixed time steps
    #folder_name_w = PATH_w * @sprintf("//full_transformation//N_%i//", N);

    # Open the file in read-write mode and overwrite the dataset
    h5open(folder_name_w * @sprintf("transformed_MAX2SAT_instance_N_%i_idx_%04i.h5", N, idx), "w") do file
        write(file, "couplings", J_mat_permuted)                    
        write(file, "local_fields", local_fields_permuted)         
        write(file, "ground_state_energy", gs_energy)        
        write(file, "time_step", time_step)
        write(file, "permutation_order", perm_order)

    end

    println("done")
end

println("============================================================ N = ", N, ", ", total_num_inst," instances, done! ============================================================")

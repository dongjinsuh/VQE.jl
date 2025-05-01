using HDF5 
using Printf
using LinearAlgebra
using DataFrames
import Random, Distributions

using VQE
using Zygote
using PyPlot
using Parameters
using NLopt
using DocStringExtensions
using OrdinaryDiffEq
#using Yao
using YaoBlocks, ChainRulesCore



# load the SK hard instance data

N = 12
#N = parse(Int, ARGS[1])

p = 15

#PATH = raw"/home/ubuntu/aqc_QAOA/SpinFluctuations.jl"
PATH = raw"/home/ubuntu/aqc_QAOA"


#subdir = "small_gaps"
# subdir = "large_gaps"

# original data
#folder_name = PATH * @sprintf("//data//N_%i//", N);

# transformation data
#folder_name = PATH * @sprintf("//transformation_hard_SK_instances//data//N_%i//", N);

# permutation data
folder_name = PATH * @sprintf("//transformation_hard_SK_instances//permutation_data//N_%i//", N);


### change N in the pattern individually 
pattern = r"transformed_hard_SK_instance_N_12_seed_(\d+)\.h5"

# original data
#pattern = r"hard_random_SK_instance_N_12_seed_(\d+)\.h5"

###

instance_names = readdir(folder_name)
loop_var = 1
#loop_var = parse(Int, ARGS[2])
total_num_inst = 0


num_params = Int((2*N/2+3)*p)

best_cost = 0
best_params = zeros(num_params)
best_initial_params = zeros(num_params)
best_probs = zeros(Float64, 2^N)
average_params = zeros(num_params)


# find optimal initial params with first 20 instances 

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+19])

    global total_num_inst = k
    println("current instance count: ", k)

    seed = match(pattern, instance_name)[1]
    seed = parse(Int64, seed)
    #seed = 102149
    println("seed: ", seed)
    #spin_idx = 2

    #file_name = folder_name * @sprintf("hard_random_SK_instance_N_%i_seed_%i.h5", N , seed)
    file_name = folder_name * @sprintf("transformed_hard_SK_instance_N_%i_seed_%i.h5", N , seed)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, local_fields, J_mat);
    println(gs_energy)

    SK_problem = Problem(p, local_fields, J_mat)

    #num_params = Int((2*N/2+3)*p)


    global learning_rate = 0.02 #0.00005  
    niter = 100

    global best_cost = 0
    global best_lr = learning_rate

    
    min_params = 0.5 #0.001 # starting initial parameter
    #max_params = 0.8
    n_params_step = 6
    n_lr_step = 3

    for j in 1:n_lr_step
        
        for i in 1:n_params_step
             
            initial_params = vcat([min_params for _ in 1:num_params])
            cost, params, probs, cost_hist = optimize_parameters_half(SK_problem, initial_params, niter=niter, learning_rate=learning_rate)
            #println(initial_params[1])
            #println(cost)
            #println(params)
            
            if abs(cost) > abs(best_cost)
                global best_cost = cost
                global best_params = params
                global best_initial_params = initial_params
                global best_probs = probs
                global best_lr = learning_rate
            end
        
            min_params += 0.1
        end    

        min_params = 0.5
        global learning_rate += 0.02 #0.002

    end

    println("Optimal cost: ", best_cost)
    println("Optimal initial parameter: ", best_initial_params[1])
    println("Optimal parameter: ", best_params[1:10])
    println(best_lr)
    
    # save the optimal parameters and cost

    PATH_w = raw"/home/ubuntu/aqc_VQE"

    folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//optimized_parameter_data_20//", N, p);

    #h5open(folder_name_w * @sprintf("generalized_angles_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
    h5open(folder_name_w * @sprintf("optimized_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
        write(file, "ground_state_energy", gs_energy)           
        write(file, "seed", seed)                               # Save seed
        write(file, "cost_value", best_cost)                    # Save cost values
        write(file, "parameter_angles", best_params)            # Save parameters
        write(file, "initial_parameters", best_initial_params)  # Save initial parameters
        write(file, "learning_rate", best_lr)                   # Save learning rate
        write(file, "probabilities", best_probs)                # Save probs

    end


    for i in 1:length(best_params)
        global average_params[i] += best_params[i]
    end

end

average_params .= average_params ./ 20
println("average parameters: ", average_params)



##########################
### optimization with the first optimized initial params

best_initial_params = copy(average_params)

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+99])

    global total_num_inst = k
    println("current instance count: ", k)

    seed = match(pattern, instance_name)[1]
    seed = parse(Int64, seed)
    println("seed: ", seed)
    #spin_idx = 2

    file_name = folder_name * @sprintf("transformed_hard_SK_instance_N_%i_seed_%i.h5", N , seed)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, local_fields, J_mat);
    println(gs_energy)


    SK_problem = Problem(p, local_fields, J_mat)

    #num_params = Int((2*N/2+3)*p)

    global best_cost = 0


    niter = 100  
    
    global learning_rate = 0.0001 #0.00001
    
    for j in 1:2
        
        cost, params, probs = optimize_parameters_half(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
        if abs(cost) > abs(best_cost)
            global best_cost = cost
            global best_params = params
            #global best_initial_params = initial_params
            global best_lr = learning_rate
            global best_probs = probs
        else
            break
        end

        global learning_rate += 0.001#0.0001

    end

    for i in 1:2

        cost, params, probs = optimize_parameters_half(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
        if abs(cost) > abs(best_cost)
            global best_cost = cost
            global best_params = params
            #global best_initial_params = initial_params
            global best_lr = learning_rate
            global best_probs = probs
        else
            break
        end

        global learning_rate += 0.02 #0.01

    end

    # one big increase step for learning rate
    global learning_rate = 0.1
    
    cost, params, probs = optimize_parameters_half(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
    if abs(cost) > abs(best_cost)
        global best_cost = cost
        global best_params = params
        #global best_initial_params = initial_params
        global best_lr = learning_rate
        global best_probs = probs
    end
    

    println("Optimal cost: ", best_cost)
    println("Optimal initial parameter: ", best_initial_params[1])
    println("Optimal parameter: ", best_params[1:10])
    println(best_lr)

    
    # save the optimal parameters and cost

    PATH_w = raw"/home/ubuntu/aqc_VQE"

    folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//average_parameter_data//", N, p);

    #h5open(folder_name_w * @sprintf("generalized_angles_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
    h5open(folder_name_w * @sprintf("optimized_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
        write(file, "ground_state_energy", gs_energy)           
        write(file, "seed", seed)                               # Save seed
        write(file, "cost_value", best_cost)                    # Save cost values
        write(file, "parameter_angles", best_params)            # Save parameters
        write(file, "initial_parameters", best_initial_params)  # Save initial parameters
        write(file, "learning_rate", best_lr)                   # Save learning rate
        write(file, "probabilities", best_probs)                # Save probs
    end
      
    
    for i in 1:length(best_params)
        global average_params[i] += best_params[i]
    end

end 


average_params .= average_params ./ 100
println("average parameters: ", average_params)



######################
### using final optimized parameter without sperate optimization process for each instance

best_initial_params = copy(average_params)

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+199])

    global total_num_inst = k
    println("current instance count: ", k)

    seed = match(pattern, instance_name)[1]
    seed = parse(Int64, seed)
    println("seed: ", seed)
    #spin_idx = 2

    file_name = folder_name * @sprintf("transformed_hard_SK_instance_N_%i_seed_%i.h5", N , seed)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, local_fields, J_mat);
    println(gs_energy)


    SK_problem = Problem(p, local_fields, J_mat)

    #num_params = Int((2*N/2+3)*p)

    global best_cost = 0
    #global best_lr = learning_rate


    # one big increase step for learning rate
    global learning_rate = 0.1
    niter = 0
    
    cost, params, probs = optimize_parameters_half(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
    if abs(cost) > abs(best_cost)
        global best_cost = cost
        global best_params = params
        #global best_initial_params = initial_params
        global best_lr = learning_rate
        global best_probs = probs
    end
    

    println("Optimal cost: ", best_cost)
    println("Optimal initial parameter: ", best_initial_params[1])
    println("Optimal parameter: ", best_params[1:10])

    
    # save the optimal parameters and cost

    PATH_w = raw"/home/ubuntu/aqc_VQE"

    folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//final_parameter_data//", N, p);

    #h5open(folder_name_w * @sprintf("generalized_angles_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
    h5open(folder_name_w * @sprintf("optimized_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
        write(file, "ground_state_energy", gs_energy)           
        write(file, "seed", seed)                               # Save seed
        write(file, "cost_value", best_cost)                    # Save cost values
        write(file, "parameter_angles", best_params)            # Save parameters
        write(file, "initial_parameters", best_initial_params)  # Save initial parameters
        write(file, "learning_rate", best_lr)                   # Save learning rate
        write(file, "probabilities", best_probs)                # Save probs
    end
        
end 



println("============================================================ N = ", N, ", ", total_num_inst," instances, done! ============================================================")

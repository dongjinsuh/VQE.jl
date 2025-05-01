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



# load the MAX2SAT instance data

N = 12
#N = parse(Int, ARGS[1])
num_clauses = 3N

p = 15

 # Path to load data
PATH = raw"/home/ubuntu/MAX2SAT/"

 # Path to write data 
#PATH_w = raw"/home/ubuntu/aqc_VQE/vqe_MAX2SAT_data"
#PATH_w = raw"/home/ubuntu/aqc_VQE/vqe_MAX2SAT_data_full"
#PATH_w = raw"/home/ubuntu/aqc_VQE/vqe_MAX2SAT_data_full_fix_time"
PATH_w = raw"/home/ubuntu/aqc_VQE/vqe_MAX2SAT_data_newest"


#subdir = "small_gaps"
# subdir = "large_gaps"

# permutation data
#folder_name = PATH * @sprintf("//MAX2SAT_transformation_instances//N_%i//", N);
#folder_name = PATH * @sprintf("//MAX2SAT_transformation_instances//N_%i//cl_60//", N);
# transformation data
#folder_name = PATH * @sprintf("//MAX2SAT_transformation_instances//full_transformation//N_%i//cl_60//", N);
# transformation fix time step data 
folder_name = PATH * @sprintf("//MAX2SAT_transformation_instances//trans_test//N_%i//cl_60//", N);


### change N in the pattern individually 
pattern = r"transformed_MAX2SAT_instance_N_12_idx_(\d{4})\.h5"

###

instance_names = readdir(folder_name)
loop_var = 1
#loop_var = parse(Int, ARGS[2])
total_num_inst = 0

num_params = (2*N+1)*p

best_cost = 0
best_lr = 0
learning_rate = 0
best_params = zeros(num_params)
best_initial_params = zeros(num_params)
best_probs = zeros(Float64, 2^N)
average_params = zeros(num_params)

# find optimal initial params with first 20 instances 

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+19])

    global total_num_inst = k
    println("current instance count: ", k)

    idx = match(pattern, instance_name)[1]
    idx = parse(Int64, idx)
    println("idx: ", idx)

    file_name = folder_name * @sprintf("transformed_MAX2SAT_instance_N_%i_idx_%04i.h5", N , idx)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, - local_fields, J_mat);
    println(gs_energy)

    MAX_problem = Problem(p, - local_fields, J_mat) # minus sign added to local fields for MAX2SAT


    #num_params = (2*N+1)*p


    global learning_rate =  0.007
    niter = 100

    global best_cost = 0
    global best_lr = learning_rate

    
    # beta = X term: N parameters 
    # gamma = Z term: N parameters
    # delta = ZZ term: 1 parameters
    n_iter_gamma = 5
    n_iter_beta = 5
    n_iter_delta = 5

    max_beta = 0.6
    min_beta = 0.2
    max_gamma = 0.4
    min_gamma = 0.0
    #max_delta = 0.4
    min_delta = 0.0

    for i in 1: n_iter_beta

        beta = (min_beta + (i-1)*(max_beta-min_beta)/n_iter_beta)
        
        for j in 1: n_iter_gamma

            for l in 1: n_iter_delta
                gamma = (min_gamma + (j-1)*(max_gamma-min_gamma)/n_iter_gamma)
                delta = l
                initial_params = vcat([beta for _ in 1:N], [gamma for _ in 1:N], delta)
                initial_params = repeat(initial_params, p)
                cost, params, probs = optimize_parameters_mix(MAX_problem, initial_params, niter=niter, learning_rate=learning_rate)
                delta += 0.1
                if abs(cost) > abs(best_cost)
                    global best_cost = cost
                    global best_params = params
                    global _params = initial_params
                    global best_probs = probs
                end

            end
    
        end

    end
    


    """
    min_params = 0.1 # starting initial parameter
    #max_params = 0.1
    n_params_step = 15
    
    for j in 1:1
        
        for i in 1:n_params_step
             
            initial_params = vcat([min_params for _ in 1:num_params])
            cost, params, probs, params_hist = optimize_parameters_mix(MAX_problem, initial_params, niter=niter, learning_rate=learning_rate)
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
        
            min_params += 0.03 #0.05
        end    

        #min_params = 0.1
        #global learning_rate += 0.01

    end
    """

    println("Optimal initial parameter: ", best_initial_params[1])

    
    # save the optimal parameters and cost

    #PATH_w = raw"/home/ubuntu/aqc_VQE"

    folder_name_w = PATH_w * @sprintf("//N_%i//p%i//optimized_parameter_data_20//", N, p);

    h5open(folder_name_w * @sprintf("optimized_MAX2SAT_instance_N_%i_idx_%04i.h5", N, idx), "w") do file
        write(file, "ground_state_energy", gs_energy)           
        write(file, "idx", idx)                               # Save idx
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

#new_params = zeros(num_params)
new_params = average_params ./ 20
println("average parameters: ", new_params)
"""
average_params = zeros(num_params)


### optimization with the first optimized initial params

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+99])

    global total_num_inst = k
    println("current instance count: ", k)

    idx = match(pattern, instance_name)[1]
    idx = parse(Int64, idx)
    println("idx: ", idx)

    file_name = folder_name * @sprintf("transformed_MAX2SAT_instance_N_%i_idx_%04i.h5", N , idx)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, - local_fields, J_mat);
    println(gs_energy)

    MAX_problem = Problem(p, - local_fields, J_mat) # minus sign added to local fields for MAX2SAT

    #num_params = (2*N+1)*p   

    global best_cost = 0
    global best_lr = learning_rate


    global best_initial_params = new_params

    niter = 100  
    
    global learning_rate = 0.001

    for j in 1:2
        
        cost, params, probs = optimize_parameters_mix(MAX_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
        if abs(cost) > abs(best_cost)
            global best_cost = cost
            global best_params = params
            #global best_initial_params = initial_params
            global best_lr = learning_rate
            global best_probs = probs
        else
            break
        end

        global learning_rate += 0.005

    end
    

    println("Optimal cost: ", best_cost)
    println("Optimal initial parameter: ", best_initial_params[1])
    println("Optimal parameter: ", best_params[1:10])
    println(best_lr)

    
    # save the optimal parameters and cost

    #PATH_w = raw"/home/ubuntu/aqc_VQE"

    folder_name_w = PATH_w * @sprintf("//N_%i//p%i//average_parameter_data//", N, p);

    #h5open(folder_name_w * @sprintf("generalized_angles_hard_SK_instance_N_%i_seed_%i.h5", N, seed), "w") do file
    h5open(folder_name_w * @sprintf("optimized_MAX2SAT_instance_N_%i_idx_%04i.h5", N, idx), "w") do file
        write(file, "ground_state_energy", gs_energy)           
        write(file, "idx", idx)                               # Save idx
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


new_params = average_params ./ 100
println("average parameters: ", new_params)

"""


### using final optimized parameter without sperate optimization process for each instance

for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+199])

    global total_num_inst = k
    println("current instance count: ", k)

    idx = match(pattern, instance_name)[1]
    idx = parse(Int64, idx)
    println("idx: ", idx)

    file_name = folder_name * @sprintf("transformed_MAX2SAT_instance_N_%i_idx_%04i.h5", N , idx)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, - local_fields, J_mat);
    println(gs_energy)

    MAX_problem = Problem(p, - local_fields, J_mat) # minus sign added to local fields for MAX2SAT


    #num_params = (2*N+1)*p   

    global best_cost = 0
    global best_lr = learning_rate



    global best_initial_params = new_params

    # one big increase step for learning rate
    global learning_rate = 0.1
    niter = 0
    
    cost, params, probs = optimize_parameters_mix(MAX_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
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

    #PATH_w = raw"/home/ubuntu/aqc_VQE"

    folder_name_w = PATH_w * @sprintf("//N_%i//p%i//final_parameter_data//", N, p);

    h5open(folder_name_w * @sprintf("optimized_MAX2SAT_instance_N_%i_idx_%04i.h5", N, idx), "w") do file
        write(file, "ground_state_energy", gs_energy)           
        write(file, "idx", idx)                               # Save seed
        write(file, "cost_value", best_cost)                    # Save cost values
        write(file, "parameter_angles", best_params)            # Save parameters
        write(file, "initial_parameters", best_initial_params)  # Save initial parameters
        write(file, "learning_rate", best_lr)                   # Save learning rate
        write(file, "probabilities", best_probs)                # Save probs
    end
        
end 



println("============================================================ N = ", N, ", ", total_num_inst," instances, done! ============================================================")

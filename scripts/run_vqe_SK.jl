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



function expand_blocks(arr::Vector{T}, n::Int) where T
    """
    Given `arr` of length p*(2n+1), split into p blocks each of length (2n+1).
    For each block:
      - After the n-th element, insert two copies of that element.
      - After the (2n)-th element, insert two copies of that element.
    The resulting block has length (2n+5). Concatenate all p expanded blocks
    and return the final expanded array of length p*(2n+5).
    """
    
    # Number of blocks
    block_len = 2n + 1
    p = length(arr) ÷ block_len
    
    # Prepare the result container (length p*(2n + 5))
    new_len = p * (2n + 5)
    new_arr = Vector{T}(undef, new_len)
    
    # We'll track position in new_arr as we go
    idx_new = 1
    
    for block_idx in 0:(p-1)
        # Identify the sub-block in arr
        start_idx = block_idx * block_len + 1
        stop_idx  = start_idx + block_len - 1
        block = arr[start_idx : stop_idx]  # length (2n+1)
        
        # 1) Copy elements 1..n
        for i in 1:n
            new_arr[idx_new] = block[i]
            idx_new += 1
        end
        
        # 2) Insert two copies of the n-th element
        new_arr[idx_new]   = block[n]
        new_arr[idx_new+1] = block[n]
        idx_new += 2
        
        # 3) Copy elements (n+1)..(2n)
        for i in (n+1):(2n)
            new_arr[idx_new] = block[i]
            idx_new += 1
        end
        
        # 4) Insert two copies of the (2n)-th element
        new_arr[idx_new]   = block[2n]
        new_arr[idx_new+1] = block[2n]
        idx_new += 2
        
        # 5) Finally, copy the last element (index 2n+1)
        new_arr[idx_new] = block[2n+1]
        idx_new += 1
    end
    
    return new_arr
end


# load the SK hard instance data

N = 14
#N = parse(Int, ARGS[1])

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
pattern = r"transformed_hard_SK_instance_N_14_seed_(\d+)\.h5"

# original data
#pattern = r"hard_random_SK_instance_N_12_seed_(\d+)\.h5"

###

instance_names = readdir(folder_name)
loop_var = 1
#loop_var = parse(Int, ARGS[2])
total_num_inst = 0

#0-110
for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+100])

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

    p = 5

    SK_problem = Problem(p, local_fields, J_mat)

    #circ = VQE.circuit(SK_problem)
    #num_params = VQE.nparameters(circ)
    num_params = (2*N+1)*p


    global learning_rate =  0.001 #0.00005  
    niter = 100

    global best_cost = 0
    global best_params = zeros(num_params)
    global best_initial_params = zeros(num_params)
    global best_lr = learning_rate
    global best_probs = zeros(Float64, 2^N)

    """
    min_params = 0.001 # starting initial parameter
    #max_params = 0.8
    n_params_step = 10
    
    for j in 1:3
        
        for i in 1:n_params_step
             
            initial_params = vcat([min_params for _ in 1:num_params])
            cost, params, probs, params_hist = optimize_parameters_mix(SK_problem, initial_params, niter=niter, learning_rate=learning_rate)
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

        min_params = 0.01
        global learning_rate += 0.01 #0.002

    end
    """


     # average
    # p5 best parameter
    global best_initial_params = [0.6390877480860438, 0.6224866255315168, 0.602938634381911, 0.6323625352098211, 0.6731964355471481, 0.6334545340950863, 0.6634080408075632, 0.6432451524013939, 0.6834951853535883, 0.6444704052712127, 0.710831160280054, 0.6605972809621211, 0.3152326491155327, 0.3074071461915337, 0.28446687796967796, 0.28146275510342783, 0.29497986162765366, 0.29150522608631463, 0.28037959376746086, 0.3046320257894916, 0.26431661506708054, 0.2707226487094296, 0.32023312592071124, 0.3044956100213693, 0.24177864939337215, 0.6270917980723815, 0.6205053682706257, 0.6001659169207743, 0.6490225915149859, 0.6117317329136389, 0.6185844427975785, 0.6381583880104843, 0.5943395288893704, 0.6354221514205355, 0.6517616885856813, 0.6380901666041415, 0.6159602398582003, 0.29597358639413834, 0.2843990921737236, 0.27555331609541195, 0.2776947859501359, 0.2851896255002372, 0.274806589563551, 0.26484519619817615, 0.2875809075122922, 0.25035831326161634, 0.2517642137813675, 0.266148837457925, 0.24638399045148698, 0.1924333746985699, 0.5851017530213087, 0.5860951344419241, 0.5872361290013325, 0.638696794338721, 0.6081857976687904, 0.6316109191454874, 0.6659325119328348, 0.608370490582136, 0.6692026152625163, 0.6825667592360887, 0.6657904140984718, 0.5796908993642204, 0.3218650440950435, 0.29172213979988587, 0.2880040471187941, 0.30280077606152694, 0.29655196738235046, 0.2883504919780603, 0.28199771364029086, 0.29925600551995013, 0.2731818586971923, 0.27285446442097555, 0.2619966711903184, 0.2644572167761156, 0.4650322529528558, 0.4698365927129524, 0.4722344520706019, 0.46503870417110676, 0.5076792825908767, 0.5386377593624214, 0.5127562799713279, 0.5416610034703891, 0.48454067794709793, 0.5403432753978787, 0.5556347323406665, 0.5030634226188204, 0.42961991733939825, 0.36934039049011513, 0.3162517953871683, 0.31636593794507295, 0.34219563042237466, 0.3219505126326446, 0.3255965148244602, 0.32264129200184355, 0.32734302269920906, 0.3181027811848382, 0.3246812721282301, 0.30063645395195304, 0.3233758125820299, 0.5622241512054089, 0.3532726849726925, 0.3668183166566528, 0.3598303146761126, 0.3706629514552937, 0.34046837113826145, 0.3305136855429566, 0.33806400060721825, 0.2674907339940056, 0.28730734688243686, 0.29865693286156936, 0.24930267471884449, 0.192358643564729, 0.36876349945905734, 0.32394094338513313, 0.32443676236819985, 0.3392809874076833, 0.32436041232976837, 0.3323742141277297, 0.3285914255978471, 0.32273045888974916, 0.3257467640986078, 0.3367447589537588, 0.31169688324378414, 0.31430108543510094, 0.7391408604257177]

    # for higher N
    global best_initial_params = expand_blocks(best_initial_params, 12)

    #global best_initial_params = 
    

    """
    #niter = 100   ## using final optimized parameter without sperate optimization process for each instance
    
    global learning_rate = 0.0001 #0.00001

    #cost, params, probs = optimize_parameters_mix(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
    
    for j in 1:2
        
        cost, params, probs = optimize_parameters_mix(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
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

    for i in 1:4

        cost, params, probs = optimize_parameters_mix(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
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
    """
    # one big increase step for learning rate
    global learning_rate = 0.1
    niter = 0
    
    cost, params, probs = optimize_parameters_mix(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
    if abs(cost) > abs(best_cost)
        global best_cost = cost
        global best_params = params
        #global best_initial_params = initial_params
        global best_lr = learning_rate
        global best_probs = probs
    end
    

    ### using final optimized parameter without sperate optimization process for each instance



    println("Optimal cost: ", best_cost)
    println("Optimal initial parameter: ", best_initial_params[1])
    println("Optimal parameter: ", best_params[1:10])
    println(best_lr)

    
    # save the optimal parameters and cost

    PATH_w = raw"/home/ubuntu/aqc_VQE"

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//trans_data//N_12//p10//optimized_parameter_data_20//");
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//trans_data//N_12//p10//average_parameter_data//");
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//trans_data//N_12//p10//final_parameter_data//");

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_12//p15//optimized_parameter_data_20//");
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_12//p15//average_parameter_data//");
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_12//p15//final_parameter_data//");
    
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//origin_data//N_12//p5//optimized_parameter_data_20//");
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//origin_data//N_12//p5//average_parameter_data//");
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//origin_data//N_12//p5//final_parameter_data//");

    folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_14//p5//n12_parameter_test_data//");


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

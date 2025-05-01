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


function expand_blocks_v2(arr::Vector{T}, n::Int) where T
    # Compute block size dynamically
    block_len = n + 3  # Each block has (N + 3) elements
    p = length(arr) ÷ block_len  # Number of layers

    # Expanded block size (2 extra elements per layer)
    new_block_len = block_len + 2  
    new_len = p * new_block_len  # Compute the new array size
    new_arr = Vector{T}(undef, new_len)  # Initialize expanded array

    # Track position in new_arr
    idx_new = 1

    for block_idx in 0:(p-1)
        # Identify the sub-block in arr
        start_idx = block_idx * block_len + 1
        stop_idx  = start_idx + block_len - 1
        block = arr[start_idx : stop_idx]  # Extract current block

        # Last element remains unchanged
        last_element = block[end]

        # Determine insertion points dynamically
        partition_size = div(block_len - 1, 2)  # First (N+2)/2 elements
        first_insert_pos = partition_size  # First half insertion index
        second_insert_pos = 2 * partition_size  # Second half insertion index

        # First half
        first_half = block[1:first_insert_pos]
        insert!(first_half, first_insert_pos + 1, first_half[end])  # Copy of last element in this half
        first_half[first_insert_pos] = first_half[first_insert_pos - 1]  # Copy previous element

        # Second half
        second_half = block[first_insert_pos+1:second_insert_pos]
        insert!(second_half, first_insert_pos + 1, second_half[end])  # Copy of last element in this half
        second_half[first_insert_pos] = second_half[first_insert_pos - 1]  # Copy previous element

        # Copy to new array
        for val in first_half
            new_arr[idx_new] = val
            idx_new += 1
        end
        for val in second_half
            new_arr[idx_new] = val
            idx_new += 1
        end
        new_arr[idx_new] = last_element # Copy the last element
        idx_new += 1
    end

    return new_arr
end


# load the SK hard instance data

N = 18
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
pattern = r"transformed_hard_SK_instance_N_18_seed_(\d+)\.h5"

# original data
#pattern = r"hard_random_SK_instance_N_12_seed_(\d+)\.h5"

###

instance_names = readdir(folder_name)
loop_var = 1
#loop_var = parse(Int, ARGS[2])
total_num_inst = 0

#0-110
for (k, instance_name) in enumerate(instance_names[loop_var+0:loop_var+99])

    global total_num_inst = k
    println("current instance count: ", k)

    seed = match(pattern, instance_name)[1]
    seed = parse(Int64, seed)
    #seed = 102149
    println("seed: ", seed)
    #spin_idx = 2

    file_name = folder_name * @sprintf("transformed_hard_SK_instance_N_%i_seed_%i.h5", N , seed)

    gs_energy = h5read(file_name, "ground_state_energy") 
    J_mat = h5read(file_name, "couplings"); 
    local_fields = h5read(file_name, "local_fields")
    mf_problem = Problem(0, local_fields, J_mat);
    println(gs_energy)

    p = 4

    SK_problem = Problem(p, local_fields, J_mat)

    #circ = VQE.circuit(SK_problem)
    #num_params = VQE.nparameters(circ)
    num_params = Int((N+3)*p)


    global learning_rate =  0.001 #0.00005  
    niter = 100

    global best_cost = 0
    global best_params = zeros(num_params)
    global best_initial_params = zeros(num_params)
    global best_lr = learning_rate
    global best_probs = zeros(Float64, 2^N)


    # partly optimized N 14
    #global best_initial_params = [1.2003129717401189, 1.2147239519743644, 1.1911835654437766, 1.1895863422247566, 1.1762701543126612, 1.1632263959794786, 1.0513426069106684, 0.7721023296858889, 1.1863958448484349, 0.7803699470111823, 0.7242684311490188, 0.8046478055966259, 0.8602112795371734, 0.7418315118995291, 1.1907561091114058, 1.1796999257423086, 0.6405606080735154, 0.78605019238214, 0.885612060282734, 0.8575723976712898, 0.8685383362592822, 0.8536852443218105, 0.8668747710335183, 0.8065545153410213, 0.7085418647659509, 0.8950128565909093, 0.7933665275171637, 0.7853747762417044, 0.8551571915365294, 0.8748462730612182, 0.7814404657961278, 0.8967649061320999, 0.8798181648634261, 1.1537906768491704, 0.4958426469803733, 0.5756752840606348, 0.5638760721552315, 0.5951273071216491, 0.5933371652437817, 0.6051935592787828, 0.6637334280739492, 1.089664196696148, 0.6447290626556156, 0.943065563344977, 0.9246330661376887, 0.9768211936652296, 0.9619407995548491, 0.9309457279885472, 0.6318401466693551, 0.6219775944297509, 1.2741542020529686, 0.3984021140923554, 0.43693131762114695, 0.43902040205005993, 0.42234041085940555, 0.42396253399190004, 0.37401697403458906, 0.43688956426790315, 0.9872465475096236, 0.3299214118764871, 0.980724954645056, 0.9560237216659715, 0.9763236096064379, 0.9674672470021097, 0.9547463026400937, 0.382866751219175, 0.3010124839913218, 1.483102142313467]

    # partly optimized N 16
    global best_initial_params = [1.2003129717401182, 1.2147239519743638, 1.1911835654437786, 1.189586342224757, 1.1762701543126612, 1.1632263959794786, 1.0513426069106666, 0.9536655029289812, 0.8512026451679319, 1.1863958448484342, 0.7803699470111813, 0.7242684311490188, 0.8046478055966259, 0.8602112795371724, 0.7418315118995289, 1.190756109111405, 1.192000355250392, 1.1940942215358579, 0.6405606080735144, 0.78605019238214, 0.8856120602827348, 0.8575723976712905, 0.8685383362592816, 0.8536852443218098, 0.8668747710335185, 0.8065545153410203, 0.7861480644289567, 0.7075870975656773, 0.8950128565909096, 0.7933665275171645, 0.7853747762417046, 0.8551571915365301, 0.874846273061219, 0.7814404657961278, 0.8967649061321006, 0.8489597052336331, 0.8769305114162149, 1.1537906768491697, 0.49584264698037295, 0.5756752840606353, 0.5638760721552315, 0.5951273071216501, 0.5933371652437814, 0.6051935592787818, 0.6637334280739483, 0.7417000726992786, 1.2275685082316885, 0.6447290626556144, 0.9430655633449768, 0.9246330661376877, 0.9768211936652303, 0.9619407995548492, 0.9309457279885472, 0.6318401466693545, 0.5811348521519334, 0.6077761214044789, 1.274154202052969, 0.39840211409235576, 0.4369313176211469, 0.43902040205005993, 0.4223404108594054, 0.4239625339918998, 0.37401697403458906, 0.4368895642679028, 0.4961742907494155, 1.0805390698632784, 0.3299214118764871, 0.9807249546450553, 0.9560237216659715, 0.9763236096064386, 0.9674672470021087, 0.954746302640093, 0.38286675121917546, 0.35594327753051963, 0.29479968128021433, 1.4831021423134692]

    # partly optimized N 18
    #global best_initial_params = [1.2003129717401182, 1.2147239519743638, 1.1911835654437788, 1.189586342224757, 1.1762701543126612, 1.1632263959794786, 1.0513426069106664, 0.9536655029289813, 0.8923797349681355, 0.7758254655923015, 1.1863958448484342, 0.7803699470111812, 0.724268431149019, 0.8046478055966257, 0.8602112795371724, 0.7418315118995288, 1.190756109111405, 1.1920003552503922, 1.1517485285396112, 1.1811629435855717, 0.6405606080735143, 0.78605019238214, 0.8856120602827346, 0.8575723976712906, 0.8685383362592816, 0.8536852443218097, 0.8668747710335187, 0.8065545153410201, 0.7861480644289568, 0.7344622519900281, 0.6381977109197743, 0.8950128565909095, 0.7933665275171646, 0.7853747762417046, 0.8551571915365301, 0.874846273061219, 0.7814404657961278, 0.8967649061321004, 0.8489597052336331, 0.795674415307692, 0.8677071724990594, 1.1537906768491697, 0.49584264698037295, 0.5756752840606352, 0.5638760721552315, 0.5951273071216502, 0.5933371652437814, 0.6051935592787817, 0.6637334280739483, 0.7417000726992785, 0.7627537152446819, 1.1962697798341915, 0.6447290626556142, 0.9430655633449769, 0.9246330661376876, 0.9768211936652305, 0.9619407995548492, 0.9309457279885474, 0.6318401466693545, 0.5811348521519334, 0.5370007825056838, 0.5886856854967469, 1.274154202052969, 0.3984021140923558, 0.4369313176211468, 0.43902040205006, 0.42234041085940544, 0.42396253399189987, 0.37401697403458906, 0.4368895642679028, 0.49617429074941544, 0.52795848558065, 1.126856372093566, 0.32992141187648716, 0.9807249546450552, 0.9560237216659715, 0.9763236096064387, 0.9674672470021088, 0.9547463026400929, 0.3828667512191755, 0.3559432775305197, 0.3475130741747948, 0.2774188885041284, 1.4831021423134694]


     # average
    # p4 best parameter n12
    #global best_initial_params = [1.2003129717401204, 1.2147239519743667, 1.1911835654437755, 1.1895863422247552, 1.1762701543126612, 1.1632263959794786, 0.8787000000000005, 1.1863958448484369, 0.7803699470111838, 0.7242684311490188, 0.8046478055966259, 0.8602112795371748, 0.7418315118995298, 0.8787000000000005, 0.6405606080735159, 0.7860501923821401, 0.8856120602827332, 0.8575723976712889, 0.8685383362592829, 0.8536852443218115, 0.8668747710335174, 0.8787000000000005, 0.8950128565909087, 0.7933665275171626, 0.7853747762417033, 0.8551571915365284, 0.8748462730612171, 0.7814404657961278, 0.8787000000000005, 1.1537906768491717, 0.495842646980374, 0.5756752840606333, 0.5638760721552312, 0.5951273071216485, 0.5933371652437823, 0.6051935592787833, 0.8787000000000005, 0.6447290626556169, 0.9430655633449778, 0.9246330661376906, 0.9768211936652271, 0.9619407995548482, 0.9309457279885475, 0.8787000000000005, 1.274154202052967, 0.3984021140923549, 0.4369313176211472, 0.43902040205005993, 0.4223404108594061, 0.42396253399190087, 0.3740169740345888, 0.8787000000000005, 0.32992141187648705, 0.9807249546450578, 0.9560237216659715, 0.9763236096064365, 0.9674672470021116, 0.9547463026400957, 0.8787000000000005, 1.483102142313463]

    
    initial_params_n = expand_blocks_v2(best_initial_params, Int(N-2))

    global best_initial_params = copy(initial_params_n)    

    """
    #niter = 100   ## using final optimized parameter without sperate optimization process for each instance
    
    #cost, params, probs = optimize_parameters_two_v2(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)

    for j in 1:1
        
        cost, params, probs = optimize_parameters_two_v2(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
        if abs(cost) > abs(best_cost)
            global best_cost = cost
            global best_params = params
            #global best_initial_params = initial_params
            global best_lr = learning_rate
            global best_probs = probs
        else
            break
        end

        global learning_rate += 0.01 

    end

    for i in 1:1

        cost, params, probs = optimize_parameters_two_v2(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
        if abs(cost) > abs(best_cost)
            global best_cost = cost
            global best_params = params
            #global best_initial_params = initial_params
            global best_lr = learning_rate
            global best_probs = probs
        else
            break
        end

        #global learning_rate += 0.02 #0.01

    end
    """
    # one big increase step for learning rate
    global learning_rate = 0.1
    niter = 0
    
    cost, params, probs = optimize_parameters_two_v2(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
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

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n12_parameter_opt_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n12_parameter_test_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n12_raw_parameter_test_data//", N, p);

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n14_parameter_opt_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n14_parameter_test_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n14_raw_parameter_test_data//", N, p);

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n16_parameter_opt_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n16_parameter_test_data//", N, p);
    folder_name_w = PATH_w * @sprintf("//vqe_mix_v2_SK_data//perm_data//N_%i//p%i//n16_raw_parameter_test_data//", N, p);

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

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

    #file_name = folder_name * @sprintf("hard_random_SK_instance_N_%i_seed_%i.h5", N , seed)
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
    num_params = (2*N+1)*p


    global learning_rate =  0.001 #0.00005  
    niter = 100

    global best_cost = 0
    global best_params = zeros(num_params)
    global best_initial_params = zeros(num_params)
    global best_lr = learning_rate
    global best_probs = zeros(Float64, 2^N)


    # partly optimized N 14
    #global best_initial_params = [1.2207633120228991, 1.2206700194134792, 1.2178613638521083, 1.1977261326018975, 1.1911959184490293, 1.2035810226639405, 1.1794816830511374, 1.122511742733376, 1.1788504548812293, 1.1864964157355011, 1.1591124679023883, 1.0805928541984005, 1.0490669573497768, 1.0155163428045428, 0.7621868448257899, 0.779651686486166, 0.7623127856256684, 0.8208941768366738, 0.8442112324962197, 0.7722681880319346, 0.8276573521811774, 0.7809200373620669, 0.8583586713054526, 0.8389850756057755, 0.8887166913604096, 0.8961354884816423, 1.1095202766653376, 1.0947361464081313, 0.5948605915907806, 0.9253494118863965, 0.9341602940928918, 0.9143238482705659, 0.9151052708466474, 0.9084489840531706, 0.9321936042078144, 0.9264781834977653, 0.8922031062140131, 0.9201609666840368, 0.9185046570504956, 0.8531802087257735, 0.7116501837241189, 0.6934179560822337, 0.6742007322793337, 0.7932689532807572, 0.7889263038067468, 0.7952142932366303, 0.8606785376656915, 0.8607160352521664, 0.7917135089635964, 0.8418877568470162, 0.831103693648298, 0.846107130703066, 0.8101450967893967, 0.8395944110880629, 0.7784807682437116, 0.785891259870517, 0.7445178497449279, 1.1011454932564708, 0.6345783145696232, 0.6418020478779399, 0.6384808510564857, 0.6582581269400497, 0.6574685816230214, 0.6767550658626378, 0.6789056850696071, 0.6543723439454098, 0.6860981558605075, 0.6737970475518282, 0.6054828057704141, 0.4921997209298401, 0.5963491082699568, 0.6061073689121286, 0.9215835995437981, 0.9195253455963648, 0.907342260155389, 0.9621084740782908, 0.9424296197293576, 0.9187319090989532, 0.9362291475580663, 0.94234563648529, 0.9366045912877469, 0.907639747477386, 0.926586977054209, 0.8982832929443147, 0.5535405316471226, 0.4932736395626307, 1.2476482567953726, 0.4684904255184159, 0.464765247482284, 0.4687589150653315, 0.44938113408232266, 0.4495696068404673, 0.39562792111018724, 0.40879235930366986, 0.3719464500473898, 0.3669658672311302, 0.34015896778946425, 0.28957064772029545, 0.21684885906147244, 0.2824062190679994, 0.2698864792420272, 0.9626476418655006, 0.9610490858129169, 0.9395183123508236, 0.9607252008454311, 0.9549004171441265, 0.9417309910066084, 0.9550314515603208, 0.9449744911604842, 0.9528109429658957, 0.9279614517759708, 0.9243459903323594, 0.8794353275952937, 0.252563269658314, 0.21313830373496143, 1.533574747380723]

    # partly optimized N 16
    global best_initial_params = [1.2207633120228973, 1.220670019413479, 1.2178613638521083, 1.1977261326018993, 1.191195918449027, 1.20358102266394, 1.1794816830511368, 1.122511742733376, 1.1788504548812286, 1.1864964157355034, 1.159112467902389, 1.0805928541984005, 1.0490669573497784, 1.0155163428045442, 1.0472725672639072, 1.0399141696560343, 0.7621868448257909, 0.779651686486167, 0.7623127856256676, 0.820894176836674, 0.8442112324962204, 0.7722681880319355, 0.8276573521811772, 0.7809200373620659, 0.8583586713054527, 0.8389850756057763, 0.8887166913604096, 0.8961354884816422, 1.1095202766653363, 1.094736146408133, 1.2028446217037112, 1.1309580658231302, 0.5948605915907806, 0.9253494118863962, 0.9341602940928911, 0.9143238482705651, 0.9151052708466473, 0.908448984053171, 0.9321936042078154, 0.9264781834977643, 0.8922031062140124, 0.9201609666840367, 0.9185046570504959, 0.8531802087257742, 0.7116501837241179, 0.6934179560822342, 0.674200732279333, 0.7062116950635886, 0.7093554702642769, 0.7932689532807582, 0.7889263038067468, 0.7952142932366314, 0.8606785376656914, 0.8607160352521671, 0.7917135089635964, 0.8418877568470154, 0.8311036936482967, 0.8461071307030663, 0.8101450967893976, 0.8395944110880633, 0.7784807682437108, 0.7858912598705173, 0.7445178497449287, 0.8557749415755787, 0.7433796388917162, 1.1011454932564688, 0.6345783145696232, 0.6418020478779388, 0.6384808510564853, 0.6582581269400494, 0.6574685816230214, 0.6767550658626368, 0.6789056850696079, 0.6543723439454107, 0.6860981558605067, 0.673797047551829, 0.6054828057704141, 0.49219972092983977, 0.5963491082699566, 0.6061073689121278, 0.6858717978391488, 0.6967715336854854, 0.9215835995437989, 0.9195253455963645, 0.9073422601553889, 0.9621084740782916, 0.9424296197293577, 0.9187319090989522, 0.9362291475580664, 0.9423456364852901, 0.936604591287747, 0.9076397474773851, 0.9265869770542089, 0.8982832929443137, 0.5535405316471226, 0.4932736395626307, 0.575184723272161, 0.4616223199678389, 1.2476482567953726, 0.4684904255184156, 0.4647652474822836, 0.46875891506533185, 0.4493811340823227, 0.4495696068404672, 0.39562792111018674, 0.4087923593036704, 0.37194645004738924, 0.36696586723113006, 0.3401589677894637, 0.28957064772029545, 0.2168488590614727, 0.2824062190679998, 0.2698864792420275, 0.3041225006989684, 0.29734568654767196, 0.9626476418655013, 0.961049085812917, 0.9395183123508228, 0.9607252008454312, 0.9549004171441263, 0.9417309910066083, 0.955031451560321, 0.9449744911604844, 0.9528109429658965, 0.9279614517759709, 0.9243459903323601, 0.8794353275952937, 0.2525632696583138, 0.2131383037349614, 0.2542009551618712, 0.19130675585824408, 1.533574747380725]

    # partly optimized N 18
    #global best_initial_params =

     # average
    # p4 best parameter n12
    #global best_initial_params = [1.2207633120229004, 1.2206700194134816, 1.2178613638521076, 1.1977261326018966, 1.1911959184490315, 1.2035810226639418, 1.1794816830511399, 1.1225117427333737, 1.178850454881231, 1.186496415735499, 1.159112467902387, 1.0805928541984005, 0.7621868448257881, 0.7796516864861652, 0.7623127856256695, 0.8208941768366729, 0.8442112324962179, 0.772268188031935, 0.8276573521811783, 0.7809200373620687, 0.8583586713054518, 0.8389850756057742, 0.8887166913604081, 0.8961354884816405, 0.594860591590781, 0.9253494118863955, 0.9341602940928934, 0.9143238482705656, 0.9151052708466457, 0.9084489840531721, 0.9321936042078152, 0.9264781834977676, 0.8922031062140143, 0.9201609666840365, 0.9185046570504963, 0.8531802087257724, 0.7116501837241209, 0.7932689532807552, 0.7889263038067464, 0.7952142932366303, 0.8606785376656909, 0.8607160352521653, 0.7917135089635968, 0.8418877568470176, 0.8311036936482982, 0.8461071307030675, 0.810145096789395, 0.8395944110880638, 0.7784807682437127, 1.1011454932564722, 0.6345783145696237, 0.6418020478779414, 0.638480851056487, 0.658258126940051, 0.657468581623021, 0.6767550658626392, 0.6789056850696065, 0.6543723439454087, 0.6860981558605083, 0.6737970475518276, 0.6054828057704135, 0.4921997209298406, 0.9215835995437983, 0.9195253455963638, 0.90734226015539, 0.962108474078289, 0.9424296197293591, 0.918731909098955, 0.9362291475580671, 0.942345636485292, 0.9366045912877473, 0.907639747477386, 0.9265869770542072, 0.8982832929443164, 1.2476482567953728, 0.46849042551841535, 0.4647652474822849, 0.4687589150653321, 0.44938113408232205, 0.44956960684046693, 0.3956279211101865, 0.4087923593036697, 0.3719464500473908, 0.3669658672311307, 0.340158967789465, 0.28957064772029484, 0.21684885906147236, 0.9626476418654988, 0.9610490858129173, 0.9395183123508228, 0.9607252008454316, 0.9549004171441255, 0.9417309910066095, 0.9550314515603223, 0.9449744911604853, 0.952810942965894, 0.9279614517759696, 0.92434599033236, 0.8794353275952954, 1.5335747473807206]

    
    # for higher N
    initial_params_n = expand_blocks(best_initial_params, N-2)

    global best_initial_params = copy(initial_params_n)

    """
    
    #niter = 100   ## using final optimized parameter without sperate optimization process for each instance
    
    global learning_rate = 0.001 #0.00001

    #cost, params, probs = optimize_parameters_mix_two(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
    
    for j in 1:1
        
        cost, params, probs = optimize_parameters_mix_two(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
        if abs(cost) > abs(best_cost)
            global best_cost = cost
            global best_params = params
            #global best_initial_params = initial_params
            global best_lr = learning_rate
            global best_probs = probs
        else
            break
        end

        global learning_rate += 0.01#0.0001

    end

    for i in 1:1

        cost, params, probs = optimize_parameters_mix_two(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
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
    
    cost, params, probs = optimize_parameters_mix_two(SK_problem, best_initial_params, niter=niter, learning_rate=learning_rate)
        
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

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n12_parameter_opt_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n12_parameter_test_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n12_raw_parameter_test_data//", N, p);

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n14_parameter_opt_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n14_parameter_test_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n14_raw_parameter_test_data//", N, p);

    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n16_parameter_opt_data//", N, p);
    #folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n16_parameter_test_data//", N, p);
    folder_name_w = PATH_w * @sprintf("//vqe_mix_SK_data//perm_data//N_%i//p%i//n16_raw_parameter_test_data//", N, p);

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

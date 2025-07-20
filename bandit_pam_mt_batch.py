'''
Contains the UCB-based implementation of PAM (dubbed BanditPAM).

This is the core algorithm.
'''
from utils_batch import *
import itertools
from sklearn.metrics import silhouette_score
import time
import sys

LOGFILE="mt_batch.txt"

def build_cost_process( args):
    '''
    function to be executed on parellel by build step
    '''
    target_idx, imgs, targets, tmp_refs, best_distances, estimates, sigmas, metric, dist_mat, return_sigma = args
    imgs = np.ndarray(imgs["shape"], dtype=imgs["dtype"], buffer=imgs["obj"].buf)
    tmp_refs = np.ndarray(tmp_refs["shape"], dtype=tmp_refs["dtype"], buffer=tmp_refs["obj"].buf)
    estimates = np.ndarray(estimates["shape"], dtype=estimates["dtype"], buffer=estimates["obj"].buf)
    sigmas = np.ndarray(sigmas["shape"], dtype=sigmas["dtype"], buffer=sigmas["obj"].buf)
    best_distances = np.ndarray(best_distances["shape"], dtype=best_distances["dtype"], buffer=best_distances["obj"].buf)
    targets = np.ndarray(targets["shape"], dtype=targets["dtype"], buffer=targets["obj"].buf)

    for idx,target in targets[target_idx]:  
        idx,mean,std = build_cost_function(idx, imgs, target, tmp_refs, best_distances,metric,dist_mat)
        estimates[idx] = mean
        if return_sigma:
            sigmas[idx] = std


def build_cost_function(idx, imgs, target, tmp_refs, best_distances,metric,dist_mat):
    
    if best_distances[0] == np.inf:
        # No medoids have been assigned, can't use the difference in loss
        costs = cost_fn(imgs, target, tmp_refs, best_distances, metric = metric, use_diff = False, dist_mat = dist_mat)
    else:
        costs = cost_fn(imgs, target, tmp_refs, best_distances, metric = metric, use_diff = True, dist_mat = dist_mat)
    
    return idx, np.mean(costs), np.std(costs) / SIGMA_DIVISOR

def build_sample_for_targets(imgs, N, targets, batch_size, best_distances, num_workers, metric = None, return_sigma = False, dist_mat = None):
    '''
    For the given targets(cnadidates), which are candidate points to be assigned as medoids
    during a build step, we compute the changes in loss they would induce
    on a subsample of batch_size reference points (tmp_refs).

    The returned value is an array of estimated changes in loss for each target.
    '''
    FUNCTION_THRESHOLD = 500
    estimates=np.zeros(len(targets))
    s_estimates, estimates= create_shm_obj(estimates)
    sigmas = np.zeros(len(targets))
    s_sigmas, sigmas = create_shm_obj(sigmas)
    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype = 'int')
    s_tmp_refs, tmp_refs = create_shm_obj(tmp_refs)
    
    splitted_target = split_list([(idx, target) for idx, target in enumerate(targets)],num_workers,FUNCTION_THRESHOLD)
    s_splitted_target, splitted_target = create_shm_obj(np.array(splitted_target))

    if len(splitted_target)>1:
        input_list = [( target_idx,imgs, s_splitted_target, s_tmp_refs, best_distances, s_estimates, s_sigmas, metric, dist_mat, return_sigma) for target_idx in range(len(splitted_target))]
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [executor.submit(build_cost_process, inp) for inp in input_list]
            wait(futures)
    else:
        build_cost_process((0,imgs, s_splitted_target, s_tmp_refs, best_distances, s_estimates, s_sigmas, metric, dist_mat, return_sigma))
 
    new_estimates = estimates.round(DECIMAL_DIGITS)[:]
    new_tmp_refs = tmp_refs[:]
    new_sigmas = sigmas.copy()
    s_estimates["obj"].close()
    s_estimates["obj"].unlink()
    s_sigmas["obj"].close()
    s_sigmas["obj"].unlink()
    s_tmp_refs["obj"].close()
    s_tmp_refs["obj"].unlink()
    s_splitted_target["obj"].close()
    s_splitted_target["obj"].unlink()
    if return_sigma:
        return new_estimates, new_sigmas, new_tmp_refs
    return new_estimates, None, new_tmp_refs


def UCB_build(args, imgs,N,  num_workers, dist_mat = None):
    '''
    Performs the BUILD step of BanditPAM. Analogous to the BUILD step of PAM,
    BanditPAM assigns the initial medoids one-by-one by choosing the point at
    each step that would lower the total loss the most. Instead of computing the
    change in loss for every other point, it estimates these changes in loss.
    '''

    ### Parameters
    metric = args.metric
    p = 1. / (N * 1000)
    num_samples = np.zeros(N)
    estimates = np.zeros(N)

    
    medoids = []
    num_medoids_found = 0
    best_distances = np.inf * np.ones(N)
    s_best_distances,best_distances= create_shm_obj(best_distances)

    for k in range(num_medoids_found, args.num_medoids):
        compute_sigma = True
        print("Mediod : ",k)

        ## Initialization
        step_count = 0
        candidates = range(N) # Initially, consider all points
        lcbs = 1000 * np.ones(N)
        ucbs = 1000 * np.ones(N)
        T_samples = np.zeros(N)
        exact_mask = np.zeros(N)
        sigmas = np.zeros(N)

        original_batch_size = 100
        base = 1 # Right now, use constant batch size

        while(len(candidates) > 0):
            
            print("Step count:", step_count, ", Candidates:", len(candidates))

            this_batch_size = int(original_batch_size * (base**step_count))

            # Find the points whose change in loss should be computed exactly,
            # because >= N reference points have already been sampled.
            compute_exactly = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))[0]

            if len(compute_exactly) > 0:
               
                estimates[compute_exactly], _, calc_refs = build_sample_for_targets(imgs,N, compute_exactly, N, s_best_distances, num_workers, metric = metric, return_sigma = False, dist_mat = dist_mat)
                lcbs[compute_exactly] = estimates[compute_exactly]
                ucbs[compute_exactly] = estimates[compute_exactly]
                exact_mask[compute_exactly] = 1
                T_samples[compute_exactly] += N
                candidates = np.setdiff1d(candidates, compute_exactly) # Remove compute_exactly points from candidates so they're bounds don't get updated below

            if len(candidates) == 0: break # The last remaining candidates were computed exactly
            # Gather more evaluations of the change in loss for some reference points
            if compute_sigma:
                sample_costs, sigmas, calc_refs = build_sample_for_targets(imgs,N, candidates, this_batch_size, s_best_distances, num_workers, metric = metric, return_sigma = True, dist_mat = dist_mat)
                compute_sigma = False
            else:
                sample_costs, _, calc_refs = build_sample_for_targets(imgs,N, candidates, this_batch_size, s_best_distances, num_workers, metric = metric, return_sigma = False, dist_mat = dist_mat)
           
            # Update running average of estimates and confidence bounce
            estimates[candidates] = \
                ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * sample_costs)) / (this_batch_size + T_samples[candidates])
            T_samples[candidates] += this_batch_size
            cb_delta = sigmas[candidates] * np.sqrt(np.log(1 / p) / T_samples[candidates])
            lcbs[candidates] = estimates[candidates] - cb_delta
            ucbs[candidates] = estimates[candidates] + cb_delta
            candidates = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) )[0]
            step_count += 1
        new_medoid = np.arange(N)[ np.where( lcbs == lcbs.min() ) ]
        new_medoid = new_medoid[0]
        
        print("New Medoid:", new_medoid)

        medoids.append(new_medoid)
        _best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric, dist_mat = dist_mat)
        best_distances[:] = _best_distances[:]
    s_best_distances["obj"].close()
    s_best_distances["obj"].unlink()
    return medoids

@timer_func
def swap_sample_for_targets(imgs,N, targets, current_medoids, batch_size, num_workers, metric = None, return_sigma = False, dist_mat = None):
    '''
    For the given targets (potential swaps) during a swap step, we compute the
    changes in loss they would induce on a subsample of batch_size reference
    points (tmp_refs) when the swap is performed.

    The returned value is an array of estimated changes in loss for each target
    (swap).
    '''
   
    orig_medoids = targets[0]
    new_medoids = targets[1]
    assert len(orig_medoids) == len(new_medoids), "Must pass equal number of original medoids and new medoids"
    swaps = list(zip(orig_medoids, new_medoids))

    k = len(current_medoids)

    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
    s_tmp_refs, _= create_shm_obj(tmp_refs)

    if return_sigma:
        estimates, sigmas = cost_fn_difference_FP1(imgs, swaps, s_tmp_refs, current_medoids, num_workers, metric = metric, return_sigma = True, dist_mat = dist_mat) 
        s_tmp_refs["obj"].close()
        s_tmp_refs["obj"].unlink()
        return estimates.round(DECIMAL_DIGITS), sigmas, tmp_refs
    else:
        estimates = cost_fn_difference_FP1(imgs, swaps, s_tmp_refs, current_medoids, num_workers, metric = metric, return_sigma = False, dist_mat = dist_mat) # NOTE: depends on other medoids too!
    s_tmp_refs["obj"].close()
    s_tmp_refs["obj"].unlink()
    return estimates.round(DECIMAL_DIGITS), None, tmp_refs

@timer_func
def UCB_swap(args, imgs, N, num_workers, init_medoids, dist_mat = None):
    '''
    Performs the SWAP step of BanditPAM. Analogous to the SWAP step of PAM,
    BanditPAM chooses medoids to swap with non-medoids by performing the swap
    that would lower the total loss the most at each step. Instead of computing
    the exact change in loss for every other point, it estimates these changes.
    '''

    metric = args.metric
    k = len(init_medoids)
    p = 1. / (N * k * 1000)
    max_iter = 1e4

    medoids = init_medoids.copy()
    best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric, dist_mat = dist_mat)
    loss = np.mean(best_distances)
    iter = 0
    swap_performed = True
    while swap_performed and iter < max_iter: 
        compute_sigma = True
        iter += 1

        candidates = np.array(list(itertools.product(range(k), range(N)))) # A candidate is a PAIR
        lcbs = 1000 * np.ones((k, N)) 
        estimates = 1000 * np.ones((k, N))
        ucbs = 1000 * np.ones((k, N))

        T_samples = np.zeros((k, N))
        exact_mask = np.zeros((k, N))

        original_batch_size = 100
        base = 1 

        step_count = 0
        
        while(len(candidates) > 0):
            
            this_batch_size = int(original_batch_size * (base**step_count))

            # Find swaps whose returns should be computed exactly, because >= N
            # reference points have already been sampled
            comp_exactly_condition = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))
            compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
            if len(compute_exactly) > 0:

                exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
                estimates[exact_accesses], _, calc_refs = swap_sample_for_targets(imgs,N, exact_accesses, medoids, N, num_workers, metric = metric, return_sigma = False, dist_mat = dist_mat)
                lcbs[exact_accesses] = estimates[exact_accesses]
                ucbs[exact_accesses] = estimates[exact_accesses]
                exact_mask[exact_accesses] = 1
                T_samples[exact_accesses] += N

                cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) )
                candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))


            if len(candidates) == 0: break # The last candidates were computed exactly

            # Gather more evaluations of the change in loss for some reference points
            accesses = (candidates[:, 0], candidates[:, 1])
            if compute_sigma:
                new_samples, sigmas, calc_refs = swap_sample_for_targets(imgs,N, accesses, medoids, this_batch_size, num_workers, metric = metric, return_sigma = True, dist_mat = dist_mat)
                sigmas = sigmas.reshape(k, N) # So that can access it with sigmas[accesses] below
                compute_sigma = False
            else:
                new_samples, _, calc_refs = swap_sample_for_targets(imgs,N, accesses, medoids, this_batch_size, num_workers, metric = metric, return_sigma = False, dist_mat = dist_mat)


            # Update running average of estimates and confidence bounce
            estimates[accesses] = \
                ((T_samples[accesses] * estimates[accesses]) + (this_batch_size * new_samples)) / (this_batch_size + T_samples[accesses])
            T_samples[accesses] += this_batch_size
            cb_delta = sigmas[accesses] * np.sqrt(np.log(1 / p) / T_samples[accesses])
            lcbs[accesses] = estimates[accesses] - cb_delta
            ucbs[accesses] = estimates[accesses] + cb_delta
            cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) ) 
            candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
            step_count += 1

        # Choose the minimum amongst all losses and perform the swap
        best_swaps = zip( np.where(lcbs == lcbs.min())[0], np.where(lcbs == lcbs.min())[1] )
        best_swaps = list(best_swaps)
        best_swap = best_swaps[0]

        performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args, dist_mat = dist_mat)
       
        if performed_or_not == "NO SWAP PERFORMED":
            break

    return medoids, iter, loss

def UCB_build_and_swap(args):
    '''
    Run the entire BanditPAM algorithm, both the BUILD step and the SWAP step
    '''
    print("----------started--------------")
    num_swaps = -1
    final_loss = -1
    dist_mat = None

    total_images = load_data(args.data)
    num_workers = int(args.num_workers)
    
    
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    s_imgs,img_array = create_shm_obj(imgs)
    built_medoids = []
    N = len(imgs)
    start_time = time.time()
    # Build mediods
    built_medoids = UCB_build(args, s_imgs, N, num_workers, dist_mat = dist_mat)
    
    print("Built medoids", built_medoids)
    build_time = time.time()
    swapped_medoids = []
  
    init_medoids = built_medoids.copy()
    swapped_medoids, num_swaps, final_loss = UCB_swap(args, s_imgs, N, num_workers, init_medoids, dist_mat = dist_mat)
    print("Final medoids\n", swapped_medoids)
    end_time = time.time()
    print("\n\nAlgorithm implementation: Multithreading\n\n")
    print("When executing unit-defined functions in parallel, A batch of units will create a separate thread and run parallely \n\n")
    print("Workers : ",num_workers)
    print("\nTime Taken for Build :",build_time-start_time,"seconds\n")
    print("Time Taken for Swap :",end_time-build_time,"seconds\n")
    print("Total Time :",end_time-start_time,"seconds\n")
    s_imgs["obj"].close()
    s_imgs["obj"].unlink()
    return imgs, built_medoids, swapped_medoids, num_swaps, final_loss

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_labels(data, medoids_points):
    medoids = data[medoids_points]
    n_samples = len(data)
    k = len(medoids)
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        distances = [euclidean_distance(data[i], medoid) for medoid in medoids]
        labels[i] = np.argmin(distances)
    
    return labels

    

if __name__ == "__main__":
    
    args = get_args(sys.argv[1:])
    sys.stdout=open(LOGFILE+args.num_workers,"w")
    images, built_medoids, swapped_medoids, num_swaps, final_loss = UCB_build_and_swap(args)
    labels = get_labels(images,swapped_medoids)
    silhouette_avg = silhouette_score(images, labels)
    print(f"Silhouette Score for the mediod clusters: {silhouette_avg}")
    sys.stdout.close()
    
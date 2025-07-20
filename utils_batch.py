import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import mnist 
import argparse
import numpy as np
import multiprocessing.shared_memory as sm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor,wait

from time import time
DECIMAL_DIGITS = 5
SIGMA_DIVISOR = 1
SEED = 42


np.random.seed(SEED)

def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 3)
    parser.add_argument('-N', '--sample_size', help = 'Sampling size of dataset', type = int, default = 700)
    parser.add_argument('-m', '--metric', help = 'Metric to use (L1 or L2)', type = str)
    parser.add_argument('-w', '--num_workers', help = 'number of workers has to be run in parellel', type = str)
    parser.add_argument('-d', '--data', help = 'mnist or random', type = str)
    args = parser.parse_args(arguments)
    return args

def split_list(arr, n, thre):
    '''
    split the job list in to batches
    '''
    length = len(arr)
    chunk_size = length // n if length>=n else n
    if chunk_size<thre:
        chunk_size=thre
    return [arr[i:i + chunk_size] for i in range(0, length, chunk_size)]

def create_shm_obj(ar):
    '''
    create shared memory object
    '''
    shm = sm.SharedMemory(create=True, size=ar.nbytes)
    # Now create a NumPy array backed by shared memory
    b = np.ndarray(ar.shape, dtype=ar.dtype, buffer=shm.buf)
    b[:] = ar[:] #copy array values to the shared memory
    return {"obj":shm,"dtype":ar.dtype,"shape":ar.shape},b

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        # print(f'Function {func.__name__!r} started') 
        result = func(*args, **kwargs)
        t2 = time()
        # print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def load_data(data):
    '''
    load data based on the input
    '''
    if data == "mnist":
        N = 70000
        m = 28
        train_images = mnist.train_images()
        test_images = mnist.test_images()
        total_images = np.append(train_images, test_images, axis = 0)
        return total_images.reshape(N, m * m)/255 
    elif data == "random":
        return np.random.rand(100000, 1000)
    else: raise ValueError
    
def cost_fn(dataset, tar_idx, ref_idx, best_distances, metric = None, use_diff = True, dist_mat = None):
    '''
    Returns the "cost" of adding the pointpoint tar as a medoid:
    distances from tar to ref if it's less than the existing best distance,
    best_distances[ref_idx] otherwise

    This is called by the BUILD step of naive PAM and BanditPAM (ucb_pam).

    Contains special cases for handling trees, both with precomputed distance
    matrix and on-the-fly computation.
    '''
    
    
    if use_diff:
        return np.minimum(d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx]) - best_distances[ref_idx]
    return np.minimum(d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx])

def d(x1, x2, metric = None):
    '''
    Computes the distance between x1 and x2. If x2 is a list, computes the
    distance between x1 and every x2.
    '''
    assert len(x1.shape) == len(x2.shape), "Arrays must be of the same dimensions in distance computation"
    if len(x1.shape) > 1:
        assert x1.shape[0] == 1, "X1 is misshapen!"
        
        if metric == "L2":
            return np.linalg.norm(x1 - x2, ord = 2, axis = 1)
        elif metric == "L1":
            return np.linalg.norm(x1 - x2, ord = 1, axis = 1)
        elif metric == "COSINE":
            return pairwise_distances(x1, x2, metric = 'cosine').reshape(-1)
        else:
            raise Exception("Bad metric specified")

    else:
        assert x1.shape == x2.shape
        assert len(x1.shape) == 1
        
        if metric == "L2":
            return np.linalg.norm(x1 - x2, ord = 2)
        elif metric == "L1":
            return np.linalg.norm(x1 - x2, ord = 1)
        elif metric == "COSINE":
            return cosine(x1, x2)
        else:
            raise Exception("Bad metric specified")
        
def swap_cost_process(args):
    '''
    Function to be run in parallel in Swap step
    '''
    d_ns,d_n_idx,imgs, tmp_refs, distinct_new_medoids, reidx_lookup, ALL_new_med_distances,metric = args
    imgs = np.ndarray(imgs["shape"], dtype=imgs["dtype"], buffer=imgs["obj"].buf)
    tmp_refs = np.ndarray(tmp_refs["shape"], dtype=tmp_refs["dtype"], buffer=tmp_refs["obj"].buf)
    distinct_new_medoids = np.ndarray(distinct_new_medoids["shape"], dtype=distinct_new_medoids["dtype"], buffer=distinct_new_medoids["obj"].buf)
    ALL_new_med_distances = np.ndarray(ALL_new_med_distances["shape"], dtype=ALL_new_med_distances["dtype"], buffer=ALL_new_med_distances["obj"].buf)
    d_ns = np.ndarray(d_ns["shape"], dtype=d_ns["dtype"], buffer=d_ns["obj"].buf)
    for idx,d_n in d_ns[d_n_idx]:
        reidx_lookup[d_n] = idx 
        ALL_new_med_distances[idx] = d(imgs[d_n].reshape(1, -1), imgs[tmp_refs], metric)
    

@timer_func
def cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, num_workers, metric = None, return_sigma = False, use_diff = True, dist_mat = None):
    '''
    Returns the new losses if we were to perform the swaps in swaps, as in
    cost_fn_difference above, but using the FastPAM1 optimization.
    '''
    FUNCTION_THRESHOLD=500

    num_targets = len(swaps)
    reference_best_distances, reference_closest_medoids, reference_second_best_distances = get_best_distances(current_medoids, imgs, subset = tmp_refs, return_second_best = True, metric = metric, dist_mat = dist_mat)
    
    new_losses = np.zeros(num_targets)
    sigmas = np.zeros(num_targets)

    distinct_new_medoids = np.array(list(set([s[1] for s in swaps])))
    s_distinct_new_medoids, distinct_new_medoids= create_shm_obj(distinct_new_medoids)
    ALL_new_med_distances = np.zeros((len(distinct_new_medoids), tmp_refs["shape"][0])) 
    s_ALL_new_med_distances, ALL_new_med_distances= create_shm_obj(ALL_new_med_distances)
    reidx_lookup = mp.Manager().dict()

    dnm_list=[(idx, target) for idx, target in enumerate(distinct_new_medoids)]
    if tmp_refs["shape"][0] < 1000:
        splitted_dnm_list = split_list(dnm_list,num_workers,FUNCTION_THRESHOLD)
    else:
        splitted_dnm_list = [[i] for i in dnm_list]
    
    s_splitted_dnm, splitted_dnm = create_shm_obj(np.array(splitted_dnm_list))
    if len(splitted_dnm)>1:
        input_list =  [(s_splitted_dnm,d_n_idx,imgs, tmp_refs, s_distinct_new_medoids, reidx_lookup, s_ALL_new_med_distances,metric) for d_n_idx in range(len(splitted_dnm))]
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [executor.submit(swap_cost_process, inp) for inp in input_list]
            wait(futures)
    else:
        swap_cost_process((s_splitted_dnm,0,imgs, tmp_refs, s_distinct_new_medoids, reidx_lookup, s_ALL_new_med_distances,metric))

    for s_idx, s in enumerate(swaps):
        
        old_medoid = current_medoids[s[0]]
        new_medoid = s[1]
        case1 = np.where(reference_closest_medoids == old_medoid)[0] # List of indices
        case2 = np.where(reference_closest_medoids != old_medoid)[0] # List of indices
        new_medoid_distances = ALL_new_med_distances[reidx_lookup[new_medoid]]
        case1_losses = np.minimum( new_medoid_distances[case1], reference_second_best_distances[case1] )
        case2_losses = np.minimum( new_medoid_distances[case2], reference_best_distances[case2] )
       
        if use_diff:
            case1_losses -= reference_best_distances[case1]
            case2_losses -= reference_best_distances[case2]

        new_losses[s_idx] = np.sum(case1_losses) + np.sum(case2_losses)

        if return_sigma:
            sigmas[s_idx] = np.std(np.hstack((case1_losses, case2_losses))) / SIGMA_DIVISOR

    new_losses /= len(tmp_refs)

    s_distinct_new_medoids["obj"].close()
    s_distinct_new_medoids["obj"].unlink()
    s_ALL_new_med_distances["obj"].close()
    s_ALL_new_med_distances["obj"].unlink()
    s_splitted_dnm["obj"].close()
    s_splitted_dnm["obj"].unlink()
    del(reidx_lookup)
    if return_sigma:
        return new_losses, sigmas
    return new_losses


@timer_func
def get_best_distances(medoids, imgs, subset = None, return_second_best = False, metric = None, dist_mat = None):
    '''
    For each point, calculate the minimum distance to any medoid.

    Do not call this from random fns which subsample the dataset, or your
    indices will be thrown off.
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    assert not (return_second_best and len(medoids) < 2), "Need at least 2 medoids to avoid infs when asking for return_second_best"
    dataset = np.ndarray(imgs["shape"], dtype=imgs["dtype"], buffer=imgs["obj"].buf)
 
    inner_d_fn = d

    if subset is None:
        N = len(dataset)
        refs = range(N)
    else:
        refs = np.ndarray(subset["shape"], dtype=subset["dtype"], buffer=subset["obj"].buf)

    best_distances = np.array([float('inf') for _ in refs])
    second_best_distances = np.array([float('inf') for _ in refs])
    closest_medoids = np.array([-1 for _ in refs])

    for p_idx, point in enumerate(refs):
        for m in medoids:
        
            if inner_d_fn(dataset[m], dataset[point], metric) < best_distances[p_idx]:
                second_best_distances[p_idx] = best_distances[p_idx]
                best_distances[p_idx] = inner_d_fn(dataset[m], dataset[point], metric)
                closest_medoids[p_idx] = m
            elif inner_d_fn(dataset[m], dataset[point], metric) < second_best_distances[p_idx]:
                second_best_distances[p_idx] = inner_d_fn(dataset[m], dataset[point], metric)
        
    if return_second_best:
        return best_distances, closest_medoids, second_best_distances
    return best_distances, closest_medoids

@timer_func
def medoid_swap(medoids, best_swap, imgs, loss, args, dist_mat = None):
    '''
    Swaps the medoid-nonmedoid pair in best_swap if it would lower the loss on
    the datapoints in imgs. Returns a string describing whether the swap was
    performed, as well as the new medoids and new loss.
    '''

    orig_medoid = medoids[best_swap[0]]
    new_medoid = best_swap[1]

    new_medoids = medoids.copy()
    new_medoids.remove(orig_medoid)
    new_medoids.append(new_medoid)
    new_best_distances, new_closest_medoids = get_best_distances(new_medoids, imgs, metric = args.metric, dist_mat = dist_mat)
    new_loss = np.mean(new_best_distances)
    performed_or_not = ''
    if new_loss < loss:
        performed_or_not = "SWAP PERFORMED"
        swap_performed = True
    else:
        performed_or_not = "NO SWAP PERFORMED"
        new_medoids = medoids

    print("Tried to swap", orig_medoid, "with", new_medoid)
    print(performed_or_not)
    print("Old loss:", loss)
    print("New loss:", new_loss)

    return performed_or_not, new_medoids, min(new_loss, loss)

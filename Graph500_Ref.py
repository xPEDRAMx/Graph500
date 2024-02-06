import numpy as np
from scipy.sparse import csr_matrix

def kronecker_generator(SCALE, edgefactor):
    """
        Generate an edgelist according to the Graph500 parameters.
        In this sample, the edge list is returned in an array with three
        rows, where StartVertex is first row, EndVertex is the second row,
        and Weight is the third row. The vertex labels start at zero.
    """
    
    # Set number of vertices.
    N = 2 ** SCALE

    # Set number of edges.
    M = int(edgefactor * N)

    # Set initiator probabilities.
    [A, B, C] = [0.57, 0.19, 0.19]

    # Create index arrays.
    ijw = np.ones((3, M))
    # Loop over each order of bit
    ab = A + B
    c_norm = C / (1 - (A + B))
    a_norm = A / (A + B)

    for ib in range(1, SCALE + 1):
        # Compare with probabilities and set bits of indices.
        ii_bit = np.random.uniform(0, 1, size = (1, M)) > ab
        jj_bit = np.random.uniform(0, 1, size = (1, M)) > (c_norm * ii_bit + a_norm * (~ii_bit))
        ijw[0:2] = ijw[0:2] + 2 ** (ib - 1) * np.append(ii_bit, jj_bit, axis = 0)

    # Generate weights.
    ijw[2] = np.random.uniform(0, 1, size = (1, M))

    # Permute vertex labels and edge list.
    ijw[0] = np.random.permutation(ijw[0])
    ijw[1] = np.random.permutation(ijw[1])

    # Adjust to zero-based labels.
    ijw[0:2] = ijw[0:2] - 1

    return ijw



####STEP KERNEL 1

def kernel_1(ijw):
    """
        Compute a sparse adjacency matrix representation
        of the graph with edges from ijw
    """

    #print(ijw)

    # Remove self-edges
    delete_index = []
    for j in range(ijw.shape[1]):
        if ijw[0][j] == ijw[1][j]:
            delete_index.append(j)
    ijw = np.delete(ijw, np.s_[delete_index], axis = 1)
    
    # Adjust away from zero labels.
    ijw[0:2,:] = ijw[0:2,:] + 1
    
    #print(ijw)

    # Order into a single triangle.
    mask = ijw[0] < ijw[1]
    for j in range(len(mask)):
        if mask[j]:
            ijw[0][j], ijw[1][j] = ijw[1][j], ijw[0][j]
        
    #print(ijw)
    
    # Find the maximum label from sizing.
    N = int(max(max(ijw[0]), max(ijw[1])))
    
    # Create the matrix, ensure it is square.
    G = np.zeros((N, N))
    for j in range(ijw.shape[1]):
        r = int(ijw[0][j] - 1)
        c = int(ijw[1][j] - 1)
        if G[r][c] != 0:
            G[r][c] = min(G[r][c], ijw[2][j])
        else:
            G[r][c] = ijw[2][j]
      
    G = G + G.T
    
    return G



#STEP KERNEL 2

def kernel_2(G, root):
    """
        Compute s sparse adjacency matrix representation
        of the graph with edges from ij.
        
        root here is zero-based label: 0 to 2^N-1
    """
    
    N = G.shape[0]

    # Not adjust from zero labels, just use it.
    parent = np.full((N, 1), -1)
    parent[root][0] = root
    
    vlist = np.full((N, 1), -1)
    vlist[0][0] = root;
    vlist = vlist.astype(int)
    lastk = 1
    for k in range(N):
        v = vlist[k][0];
        if v == -1: break
        nxt_candidate = np.where(G[:, v]!=0)[0]
        for neighbor in nxt_candidate: 
            if parent[neighbor][0] == -1:
                parent[neighbor][0] = v
                vlist[lastk][0] = neighbor
                lastk += 1
    
    # Adjust to zero labels
    #parent = parent - 1
    
    return parent



####OUTPUT

def output(SCALE, NBFS, NSSSP, kernel_1_time, kernel_2_time, kernel_2_nedge):

    print("SCALE: %d" % SCALE)
    print("NBFS: %d" % NBFS)
    print("construction_time: %20.17e\n" % kernel_1_time)
    
    print("bfs_min_time: %20.17e" % np.percentile(kernel_2_time, 0))
    print("bfs_firstquartile_time: %20.17e" % np.percentile(kernel_2_time, 25))
    print("bfs_median_time: %20.17e" % np.percentile(kernel_2_time, 50))
    print("bfs_thirdquartile_time: %20.17e" % np.percentile(kernel_2_time, 75))
    print("bfs_max_time: %20.17e" % np.percentile(kernel_2_time, 100))
    print("bfs_mean_time: %20.17e" % np.mean(kernel_2_time))
    print("bfs_stddev_time: %20.17e\n" % np.std(kernel_2_time))
    
    TEPS = kernel_2_nedge / kernel_2_time
    print("TEPS: %20.17e" % np.mean(TEPS))

####

def validate(parent, G):
    for i in range(len(parent)):
        if G[i][parent[i]] == 0:
            return False
    return True


####


import time


"""
    Driver, not include kernel_3
"""

SCALE = 11
edgefactor = 16
NBFS = 64

ijw = kronecker_generator(SCALE, edgefactor)

start1 = time.time()

G = kernel_1(ijw);
end1 = time.time()
kernel_1_time = end1 - start1

N = G.shape[0]

#print(G)

# Find all node labels that are not isolated in graph.
valid_node = np.array(np.where(G.any(axis=0)))
search_key = np.random.permutation(valid_node[0])
if len(search_key) > NBFS:
    search_key = search_key[0: NBFS + 1]
else:
    NBFS = len(search_key)
# Search keys are already zero-based

kernel_2_time = np.full((NBFS, 1), np.inf)
kernel_2_nedge = np.zeros((NBFS, 1))

"""
k = 0 : NBFS-1 in python, because it is the index of list,
search_key itself as a list includes zero-based node labels.
"""
for k in range(NBFS):
    t_start = time.time()
    parent = kernel_2(G, search_key[k])
    t_end = time.time()
    kernel_2_time[k] = t_end - t_start
    
    # Validation 
    
    if not validate(parent, G):
        print("BFS from search key %d failed to be validated" % search_key[k])
    
    for node in parent:
        if node>=0:
            kernel_2_nedge[k] += len(np.where(G[:,node]>0)[0])
            
    # kernel_3 ignored
    
output (SCALE, NBFS, NBFS, kernel_1_time, kernel_2_time, kernel_2_nedge)
print (output)

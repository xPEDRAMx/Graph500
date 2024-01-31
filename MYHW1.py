import numpy as np
from scipy.sparse import csr_matrix
import time

def kronecker_generator(SCALE, edgefactor):
    """
    Generate an edgelist according to the Graph500 parameters.
    In this sample, the edge list is returned in an array with three
    rows, where StartVertex is the first row, EndVertex is the second row,
    and Weight is the third row. The vertex labels start at zero.
    """
    N = 2 ** SCALE
    M = int(edgefactor * N)
    [A, B, C] = [0.57, 0.19, 0.19]

    ijw = np.ones((3, M))
    ab = A + B
    c_norm = C / (1 - (A + B))
    a_norm = A / (A + B)

    for ib in range(1, SCALE + 1):
        ii_bit = np.random.uniform(0, 1, size=(1, M)) > ab
        jj_bit = np.random.uniform(0, 1, size=(1, M)) > (c_norm * ii_bit + a_norm * (~ii_bit))
        ijw[0:2] = ijw[0:2] + 2 ** (ib - 1) * np.append(ii_bit, jj_bit, axis=0)

    ijw[2] = np.random.uniform(0, 1, size=(1, M))

    ijw[0] = np.random.permutation(ijw[0])
    ijw[1] = np.random.permutation(ijw[1])

    ijw[0:2] = ijw[0:2] - 1

    return ijw

def kernel_1(ijw):
    """
    Compute a sparse adjacency matrix representation
    of the graph with edges from ijw
    """
    delete_index = []
    for j in range(ijw.shape[1]):
        if ijw[0][j] == ijw[1][j]:
            delete_index.append(j)
    ijw = np.delete(ijw, np.s_[delete_index], axis=1)

    ijw[0:2, :] = ijw[0:2, :] + 1

    mask = ijw[0] < ijw[1]
    for j in range(len(mask)):
        if mask[j]:
            ijw[0][j], ijw[1][j] = ijw[1][j], ijw[0][j]

    N = int(max(max(ijw[0]), max(ijw[1])))
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

class Graph:
    def BFS(self, start_node):
        result = []  # Array to store the BFS result
        queue = [start_node]
        visited = set([start_node])

        while queue:
            s = queue.pop(0)
            result.append(s)  # Append the visited node to the result array

            for neighbor in np.where(self.graph[s] > 0)[0]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

        return np.array(result)

def output(SCALE, NBFS, NSSSP, kernel_1_time, kernel_2_time, kernel_2_nedge):
    import numpy as np

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

# Example usage:
SCALE = 11
edgefactor = 16
NBFS = 64

ijw_result = kronecker_generator(SCALE, edgefactor)

start1 = time.time()
G_result = kernel_1(ijw_result)
end1 = time.time()
kernel_1_time = end1 - start1

G = Graph()
G.graph = G_result

kernel_2_time = np.full((NBFS, 1), np.inf)
kernel_2_nedge = np.zeros((NBFS, 1))

search_key = np.random.permutation(range(len(G_result)))[0:NBFS]

for k in range(NBFS):
    t_start = time.time()
    parent = G.BFS(search_key[k])
    t_end = time.time()
    kernel_2_time[k] = t_end - t_start

    for node in parent:
        if node >= 0:
            kernel_2_nedge[k] += len(np.where(G_result[:, node] > 0)[0])

output(SCALE, NBFS, NBFS, kernel_1_time, kernel_2_time, kernel_2_nedge)

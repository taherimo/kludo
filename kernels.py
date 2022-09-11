import numpy as np
from scipy.linalg import expm
import math
import multiprocessing as mp
from itertools import repeat

n_cores = mp.cpu_count()

def reg_lap_kernel(graph, alpha): # aka normalized random walk with restart kernel

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    lap_matrix = deg_matrix - adj_matrix

    kernel = np.linalg.inv(np.identity(np.shape(lap_matrix)[0]) + alpha * lap_matrix)


    return kernel

def markov_exp_diff_kernel(graph, beta):

    # num_vtx = len(graph.vs)
    #
    # adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    # deg_matrix = np.diag(graph.strength(weights='weight'))
    # m_matrix = (deg_matrix - adj_matrix - num_vtx * np.identity(np.shape(adj_matrix)[0]))/num_vtx

    # norm_factor = graph.maxdegree() + 1
    norm_factor = max(graph.strength(weights='weight')) + 1

    lap_matrix = graph.laplacian(weights='weight')
    m_matrix = (lap_matrix - norm_factor * np.identity(np.shape(lap_matrix)[0])) / norm_factor

    kernel = expm(-1 * beta * m_matrix)

    return kernel

def markov_diff_kernel(graph, t):

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(np.array(graph.strength(weights='weight'))+0.00001)
    p_matrix = np.matmul(np.linalg.inv(deg_matrix), adj_matrix)

    p_matrix_power = p_matrix.copy()
    z_matrix = p_matrix.copy()

    for tau in range(2, int(t + 1), 1):
        p_matrix_power = np.dot(p_matrix_power, p_matrix)
        z_matrix += p_matrix_power

    z_matrix = z_matrix / t

    # pool = mp.Pool(n_cores)
    # out = pool.starmap(np.linalg.matrix_power, zip(repeat(p_matrix),list(np.arange(1, int(t + 1), step=1))))
    # z_matrix = np.sum(out, axis=0) / t

    kernel = np.matmul(z_matrix, z_matrix.transpose())


    return kernel

def lap_exp_diff_kernel(graph, beta):

    lap_matrix = np.array(graph.laplacian(weights='weight'))

    # adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    # deg_matrix = np.diag(graph.strength(weights='weight'))
    # lap_matrix = deg_matrix - adj_matrix

    kernel = expm(-1 * beta * lap_matrix)

    return kernel


# def exp_diff_kernel(graph, alpha):
#     adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
#     kernel = expm(alpha * adj_matrix)
#     return kernel


def calc_kernel(graph, type, bw):
    kernel_matrix = None
    if type == 'LED':
        kernel_matrix = lap_exp_diff_kernel(graph, bw)
    elif type == 'MD':
        kernel_matrix = markov_diff_kernel(graph, bw)
    elif type == 'MED':
        kernel_matrix = markov_exp_diff_kernel(graph, bw)
    elif type == 'RL':
        kernel_matrix = reg_lap_kernel(graph, bw)
    return kernel_matrix



def convert_kernel_to_distance(kernel_matrix, method='norm'):
    n = kernel_matrix.shape[0]
    distance_matrix = np.zeros((n, n))

    if method == 'norm':
        for i in range(n - 1):
            for j in range(i + 1, n):
                x = kernel_matrix[i,i] - 2 * kernel_matrix[i,j] + kernel_matrix[j,j]
                if x < 0:
                    x = 0
                distance_matrix[i,j] = distance_matrix[j,i] = math.sqrt(x)

        return distance_matrix

    elif method == 'CS':
        for i in range(n - 1):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = math.acos(kernel_matrix[i, j] ** 2) / (kernel_matrix[i,i] * kernel_matrix[j,j])

        return distance_matrix

    elif method == 'exp':
        for i in range(n - 1):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = math.exp(-1 * kernel_matrix[i, j])

        return distance_matrix

    elif method == 'naive':
        for i in range(n - 1):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = 1 - (kernel_matrix[i,j] / math.sqrt(kernel_matrix[i,i] * kernel_matrix[j,j]))

        return distance_matrix


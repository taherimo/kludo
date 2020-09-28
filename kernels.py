import numpy as np
from scipy.linalg import expm
import math

def reg_lap_kernel(graph, alpha): # aka normalized random walk with restart kernel

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    lap_matrix = deg_matrix - adj_matrix

    kernel = np.linalg.inv(np.identity(np.shape(lap_matrix)[0]) + alpha * lap_matrix)


    return kernel

def markov_exp_diff_kernel(graph, beta):

    num_vtx = len(graph.vs)

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    m_matrix = (deg_matrix - adj_matrix - num_vtx * np.identity(np.shape(adj_matrix)[0]))/num_vtx

    kernel = expm(-1 * beta * m_matrix)


    return kernel

def markov_diff_kernel(graph, t):

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    p_matrix = np.zeros(shape=(np.shape(adj_matrix)[0],np.shape(adj_matrix)[1]))

    for i in range (np.shape(adj_matrix)[0]):
        for j in range (np.shape(adj_matrix)[1]):
            if deg_matrix[i,i] > 0:
                p_matrix[i,j]= adj_matrix[i,j] / deg_matrix[i,i]
            elif deg_matrix[i,i] == 0:
                p_matrix[i, j] = adj_matrix[i, j] / 0.00001

    z_matrix = np.zeros(shape=(np.shape(adj_matrix)[0],np.shape(adj_matrix)[0]))

    for tau in np.arange(1, int(t + 1), step=1):
        z_matrix = z_matrix + np.linalg.matrix_power(p_matrix, tau)
    z_matrix = z_matrix / t

    kernel = np.matmul(z_matrix, z_matrix.transpose())


    return kernel

def lap_exp_diff_kernel(graph, beta):

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    lap_matrix = deg_matrix - adj_matrix

    kernel = expm(-1 * beta * lap_matrix)

    return kernel


def convert_kernel_to_distance(kernel_matrix, method):
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


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sys import exit
from utils import *
from math import ceil

import multiprocessing as mp
import time

DISCLAIMER = -1
rg = np.random.default_rng()

start_time = time.time()

nvars = 2  # number of choice variants
# ----------------------------------------------------------
# community specification
# ----------------------------------------------------------

# specify community net
net = nx.complete_graph(250)
setattr(net, 'nvars', nvars)  # associte 'nvars' with 'net'

# set parameters of community actors
for n in net:
    net.nodes[n]['rho'] = 20
    if n == 0:
        net.nodes[n]['choice'] = 0
    else:
        net.nodes[n]['choice'] = DISCLAIMER

# set parameters of community channels
for channel in net.edges:
    alice = min(channel)
    if alice == 0:
        net.edges[channel]['a'] = 1.0
        net.edges[channel]['D'] = define_dialogue_matrix(
            1.0,
            rg.uniform(low=0.2, high=0.6)
        )
    else:
        net.edges[channel]['a'] = 1.0
        net.edges[channel]['D'] = define_dialogue_matrix(
            rg.uniform(low=0.2, high=0.6),
            rg.uniform(low=0.2, high=0.6)
        )

print(f"Initialization : {time.time() - start_time}")


def simulate_dialog(alice, bob):
    global net
    D = net.edges[alice, bob]['D']
    wA = net.nodes[alice]['w']
    wB = net.nodes[bob]['w']
    wA_result = np.zeros(net.nvars)
    for v in range(net.nvars):
        wA_result[v] = D[0, 0] * wA[v] + D[0, 1] * wB[v]
    return wA_result


def group(xs, parts):
    """
    Helper function to split a list into chunks
    :param xs:    - input list
    :param parts: - number of chunks
    :return:      - generator with chunks
    """
    part_len = ceil(len(xs) / parts)
    return (xs[part_len * k:part_len * (k + 1)] for k in range(parts))


def simulate_group_of_dialogs(vertex, n):
    """
    Simulates all possible dialogues between Vertex and its neighbours
    :param vertex: - a particular number of vertex to work on
    :param n: - aggregate number of vertexes
    :return: - pair Vertex and Resulting value
    """
    global net, brntrial_res
    # Allows to check that there is only one copy of net
    # print(f"addr of the net is : {hex(id(net))}")

    total = np.zeros(2)
    counter = 0
    for vertex_i in range(n):
        alice, bob = min(vertex, vertex_i), max(vertex, vertex_i)
        if vertex_i != vertex and \
                brntrial_res[alice, bob]:
            np.add(total, simulate_dialog(alice, bob), total)
            counter += 1

    return [vertex, total / max(counter, 1)]


def calc_brntrial(v, n):
    """ calculates bernoulli trial results for a particular vertex
    :param v: a particular vertex
    :param n: number of nodes
    :return: dict of corresponding results
    """
    return {
        (v, v_i): Bernoulli_trial(net.edges[v, v_i]['a'])
        for v_i in range(v + 1, n)
    }


brntrial_res = {}


@logtime
def simulate_session():
    """ Unfortunately, significant difference in performance isn't obtained.
    Parallelism is performed over multiprocess computations on vertexes, each
    process is given a vertex number and computes its dialogues with every
    relevant neighbour. We consider the directed graph.
    We avoid final bookkeeping and aggregation of results at the expanse of
    second computations of every dialogue per session(e.g. (A, B) and (B, A))"""
    global net, brntrial_res
    # Calculate bernoulli trials for the session
    if __name__ == '__main__':
        pool = mp.Pool(mp.cpu_count())
        brntrial_chunks = pool.starmap(
            calc_brntrial, (
                (vertex, len(net.nodes)) for vertex in range(len(net.nodes))
            )
        )
        pool.close()
        pool.join()

        for chunk in brntrial_chunks:
            for key, value in chunk.items():
                brntrial_res[key] = value

    # Compute dialogs
    if __name__ == "__main__":
        # pool initialization takes around 0.04
        pool = mp.Pool(mp.cpu_count())

        results = pool.starmap(
            simulate_group_of_dialogs, (
                (vertex, len(net.nodes)) for vertex in range(len(net.nodes))
            )
        )

        pool.close()
        pool.join()

        for res in results:
            net.nodes[res[0]]['w'] = res[1]


def observation():
    for n in net:
        hn = h(net.nodes[n]['w'])
        if Bernoulli_trial(
                np.power(hn, net.nodes[n]['rho'])):
            net.nodes[n]['choice'] = DISCLAIMER
        else:
            net.nodes[n]['choice'] = np.random.choice(
                net.nvars, p=net.nodes[n]['w']
            )
    # compute average preference density
    W = np.zeros(net.nvars)
    for n in net:
        np.add(W, net.nodes[n]['w'], W)
    np.multiply(W, 1.0 / net.number_of_nodes(), W)
    # compute polling result
    DP = len([1 for n in net
              if net.nodes[n]['choice'] == DISCLAIMER])
    if DP == net.number_of_nodes():
        # all community actors disclaimed a choice
        return W, 1.0, uncertainty(net.nvars)
    NP = net.number_of_nodes() - DP
    WP = net.nvars * [None]
    for v in range(net.nvars):
        WP[v] = len([1 for n in net
                     if net.nodes[n]['choice'] == v])
        WP[v] /= NP
    DP /= net.number_of_nodes()
    return W, DP, WP


# ----------------------------------------------------------
# experiment specification
# ----------------------------------------------------------
# specify initial prefernce densities of community actors
for n in net:
    if n == 0:
        net.nodes[n]['w'] = np.array([1.0, 0.0], float)
    elif n == 1:
        net.nodes[n]['w'] = np.array([0.0, 1.0], float)
    else:
        net.nodes[n]['w'] = uncertainty(net.nvars)

niter = 100  # define number of iterations

# set up the experiment

protocol = [observation()]
for istep in range(niter):
    simulate_session()
    protocol.append(observation())

print(f"Finished : {time.time() - start_time}")
# ----------------------------------------------------------
# store the experiment outcomes
# ----------------------------------------------------------
out_file = open("protocol.dat", "w")
# out_file.write(str(net.nvars) + "\n")

# output is slightly optimized by avoiding excessive file.write repetitions
out_file.write(
    '\n'.join(
        (
            ' '.join(map(str, item[0])) + ' ' +
            str(item[1]) + ' ' +
            ' '.join(map(str, item[2]))
            for item in protocol)
    )
)
out_file.close()
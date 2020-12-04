import numpy as np
import matplotlib.pyplot as plt
import copy
import tqdm
import parser


## TODO: implement 2-opt. Temporary swap function
def two_opt(nodes):
    a = np.random.randint(len(nodes)-1)
    b = np.random.randint(len(nodes)-1)
    temp = nodes[a].copy()
    nodes[a] = nodes[b]
    nodes[b] = temp
    return nodes


def distance(node1, node2):
    return np.sqrt((node1[-2]-node2[-2])**2 + (node1[-1]-node2[-1])**2)


def p_boltzmann(nodes, nodes_cand, t):
    return np.exp(-(total_distance(nodes_cand) - total_distance(nodes)) / t)


def total_distance(nodes):
    total = 0
    for i in range(len(nodes)-1):
        total += distance(nodes[i], nodes[i+1])
    return total


def main():

    # set seed for np.random module
    np.random.seed()

    # parse tsp.txt input file to node
    fname = "data/eil51.tsp.txt"
    nodes = parser.parse_file(fname, strip_node_num=True)

    # create random initial solution
    np.random.shuffle(nodes)

    # length of the markov chain
    markov_length = len(nodes)

    print("Initial distance: {}".format(total_distance(nodes)))

    # outer loop, decreasing temp
    for t in tqdm.tqdm(np.linspace(5000, 50, 100)):

        for _ in range(markov_length):
            # generate candidate solution using 2-opt
            nodes_cand = two_opt(nodes.copy())

            # y better than x, else x better than y
            if total_distance(nodes_cand) < total_distance(nodes):
                nodes = nodes_cand.copy()
            else:
                p = p_boltzmann(nodes, nodes_cand, t)
                r = np.random.random()

                # if r < p, still take new solution even if it's worse
                if r < p:
                    nodes = nodes_cand.copy()

    print("Minimum found distance: {}".format(total_distance(nodes)))

    return


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import copy
import tqdm
import parser


## TODO: implement 2-opt. Temporary swap function
def two_opt(nodes):
    """2 opt

    Args:
        nodes : np array
             array of x and y coords of a solution

    Returns:
        nodes : np array
             array of x and y coords of a solution
    """

    # randomly select 2-opt swap interval
    i = np.random.randint(0, len(nodes)-2)
    j = np.random.randint(i+1, len(nodes)-1)

    # peform 2 opt swap
    nodes[i:j] = np.flipud(nodes[i:j])

    return nodes


def swap(nodes):
    """swap 2 cities

    Args:
        nodes : np array
             array of x and y coords of a solution

    Returns:
        nodes : np array
             array of x and y coords of a solution
    """
    a = np.random.randint(len(nodes)-1)
    b = np.random.randint(len(nodes)-1)
    temp = nodes[a].copy()
    nodes[a] = nodes[b]
    nodes[b] = temp

    return nodes


def distance(node1, node2):
    """Calculate Euclidean distance between node1 and node2

    Args:
        node1 : list [x, y]
             coords of node1
        node2 : list [x, y]
             coords of node2

    Returns:
        distance : float
            Euclidean distance between node1 and node2
    """
    return np.sqrt((node1[-2]-node2[-2])**2 + (node1[-1]-node2[-1])**2)


def p_boltzmann(nodes, nodes_cand, t):
    """Calculate boltzmann distribution

    Args:
        nodes : np array
             array of x and y coords of current solution
        nodes_cand : np array
             array of x and y coords of new candidate solution
        t : float
            temperature

    Returns:
        boltzmann_value : float
            value of the boltzmann distribution
    """
    if (total_distance(nodes_cand) - total_distance(nodes)) < 0:
        print("weird")
    return np.exp(-(total_distance(nodes_cand) - total_distance(nodes)) / t)


def total_distance(nodes):
    """Calculate total distance for the tsp in solution nodes

    Args:
        nodes : np array
             array of x and y coords of a solution

    Returns:
        distance : float
            total distance between route in nodes
    """
    total = 0
    for i in range(len(nodes)-1):
        total += distance(nodes[i], nodes[i+1])
    return total


def t_schedule(curr_iter, t0):
    """Cooling schedule

    Args:
        curr_iter : int
            current iteration
        t0 : float
            initial temperature

    Returns:
        t : float
            temperature
    """
    return t0 / (1 + np.log(1 + curr_iter))


def calc_dist_opt_tour(fname_opt_tour, fname_tsp):
    """Calculates distance of given optimal tour file (opt.tour.txt)

    Args:
        fname_opt_tour : string
             file name of given optimal solution (opt.tour.txt)
        fname_tsp : string
             file name of given problem (tsp.txt)

    Returns:
        min_dist : float
             minimal distance according to optimal solution
    """
    # parse sol file to get the coordinates
    nodes = parser.get_coords_opt_tour(fname_opt_tour, fname_tsp)

    # calculate the minimal distance
    min_dist = total_distance(nodes)
    print("Minimum distance given solution: {}".format(min_dist))

    return min_dist


def main():

    fname_opt_tour = "data/eil51.opt.tour.txt"
    fname_tsp = "data/eil51.tsp.txt"

    # calculate distance for given best solution (opt.tour.txt files)
    calc_dist_opt_tour(fname_opt_tour, fname_tsp)

    # set seed for np.random module
    np.random.seed()

    # parse tsp.txt input file to node
    nodes = parser.parse_file(fname_tsp, strip_node_num=True)

    # create random initial solution
    np.random.shuffle(nodes)

    # length of the markov chain
    markov_length = len(nodes)
    t_min = 1
    t0 = 10
    t = t0
    curr_iter = 0
    print("Initial distance: {}".format(total_distance(nodes)))

    # outer loop, decreasing temp
    while t > t_min:
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

        # calculate new t
        t = t_schedule(curr_iter, t0)
        curr_iter += 1

    print("Minimum found distance: {}".format(total_distance(nodes)))

    return


if __name__ == '__main__':
    main()

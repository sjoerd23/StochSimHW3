import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import sys
import tqdm
import numba
import parser


@numba.njit
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
    i = np.random.randint(0, len(nodes)-1)
    j = np.random.randint(0, len(nodes)-1)

    # peform 2 opt swap
    if i <= j:
        nodes[i:j] = np.flipud(nodes[i:j])
    else:
        nodes = np.concatenate((nodes[j:i+1], np.flipud(nodes[0:j]),
            np.flipud(nodes[i+1:len(nodes)])))

    return nodes


@numba.njit
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


@numba.njit
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


@numba.njit
def p_boltzmann(nodes_tot_distance, nodes_cand_tot_distance, t):
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
    return np.exp(-(nodes_cand_tot_distance - nodes_tot_distance) / t)


@numba.njit
def tot_distance_sq(nodes):
    """Calculate total squared Euclidean distance for the tsp in solution nodes (for comparison of distances,
    it is unnecessary to take square root. Inequalities will still hold without)

    Args:
        nodes : np array
             array of x and y coords of a solution

    Returns:
        total_sq : float
            total squared distance between route in nodes
    """
    total_sq = (nodes[-1][-2]-nodes[0][-2])**2 + (nodes[-1][-1]-nodes[0][-1])**2
    for i in range(len(nodes)-1):
        total_sq += (nodes[i][-2]-nodes[i+1][-2])**2 + (nodes[i][-1]-nodes[i+1][-1])**2

    return total_sq


@numba.njit
def tot_distance(nodes):
    """Calculate total Euclidean distance for the tsp in solution nodes

    Args:
        nodes : np array
             array of x and y coords of a solution

    Returns:
        total : float
            total distance between route in nodes
    """
    total = np.sqrt((nodes[-1][-2]-nodes[0][-2])**2 + (nodes[-1][-1]-nodes[0][-1])**2)
    for i in range(len(nodes)-1):
        total += np.sqrt((nodes[i][-2]-nodes[i+1][-2])**2 + (nodes[i][-1]-nodes[i+1][-1])**2)

    return total


@numba.njit
def t_log(curr_iter, t0):
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


@numba.njit
def t_over_quadr(curr_iter, t0):
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
    a=0.0000000005
    return t0 / (1+a * curr_iter**2)


@numba.jit
def t_trigonometric(curr_iter, t0, tn, n):
    return tn + 0.5*(t0 - tn)*(1+np.cos((curr_iter*np.pi)/n))


def draw(nodes, title = "", ticks = True):
    """Draw the route given by notes

    Args:
        nodes : np array
             array of x and y coords of solution
        title : string
            title of plot
        ticks : boolean
            show ticks of axis of plot

    Returns:

    """
    plt.figure()
    if title:
        plt.title(title)
    x = [i[-2] for i in nodes]
    y = [i[-1] for i in nodes]
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, color='blue', zorder=2)
    plt.scatter(x, y, color='red', zorder=1)
    if not ticks:
        plt.xticks([])
        plt.yticks([])

    return


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
    min_dist = tot_distance(nodes)

    return min_dist


@numba.njit
def simulated_annealing(nodes, markov_length, t0, t_min):
    """Simulated annealing algorithm

    Args:
        nodes : np array
             array of x and y coords of a given initial solution
        markov_length : int
             length of the markov chain
        t0 : float
            initial temperature
        t_min : float
            stopping condition for temperature

    Returns:
        nodes : np array
             array of x and y coords of final solution
        nodes_shortest : np array
             array of x and y coords of best solution
    """
    t = t0
    curr_iter = 0
    # progress_counter = 0

    while t > t_min:

        # if progress_counter % 1000 == 0:
        #     print("t: ", t)
        #     progress_counter = 0
        # progress_counter += 1

        # inner loop, ov    er the Markov chain
        for _ in range(markov_length):

            # generate candidate solution using 2-opt
            nodes_cand = two_opt(nodes.copy())

            # calculate objective function
            nodes_tot_distance = tot_distance(nodes)
            nodes_cand_tot_distance = tot_distance(nodes_cand)

            # y better than x, else x better than y
            if nodes_cand_tot_distance < nodes_tot_distance:
                nodes = nodes_cand.copy()
            else:
                p = p_boltzmann(nodes_tot_distance, nodes_cand_tot_distance, t)
                r = np.random.random()

                # if r < p, still take new solution even if it's worse
                if r < p:
                    nodes = nodes_cand.copy()

        # calculate new temperature
        # t = t_log(curr_iter, t0)
        tn = t_min - 0.01
        n = 50000
        t = t_trigonometric(curr_iter, t0, tn, n)
        curr_iter += 1

    return nodes


def main():

    fname_opt_tour = "data/eil51.opt.tour.txt"
    fname_tsp = "data/eil51.tsp.txt"
    # fname_opt_tour = "data/a280.opt.tour.txt"
    # fname_tsp = "data/a280.tsp.txt"
    # fname_opt_tour = "data/pcb442.opt.tour.txt"
    # fname_tsp = "data/pcb442.tsp.txt"

    time_start = time.time()

    # set seed for np.random module
    np.random.seed()

    # calculate distance for given best solution (opt.tour.txt files)
    optimal_distance = calc_dist_opt_tour(fname_opt_tour, fname_tsp)

    # parse tsp.txt input file to nodes
    nodes = parser.parse_file(fname_tsp, strip_node_num=False)

    # specify parameters for SA
    markov_length = len(nodes)    # taken as len(nodes), can be adjusted to anything else
    t_min = 2.2
    t0 = 21

    # perform simulated annealing algorithm for a number of runs
    n_runs = 10                           # number of runs of SA algorithm
    solns = []                       # list of final solution per run
    for i in tqdm.tqdm(range(n_runs)):

        # create random initial solution
        np.random.shuffle(nodes)
        initial_distance = tot_distance(nodes)

        # perform simulated annealing
        sa = simulated_annealing(nodes, markov_length, t0, t_min)
        solns.append(sa)

    # save best solution
    distances = [tot_distance(soln) for soln in solns]
    shortest_distance = np.min(distances)
    nodes_shortest = solns[np.where(distances == shortest_distance)[0][0]]

    # node number has to be parsed to be able to save
    if len(solns[0][0]) == 3:
        fname_nodes_shortest = "results/nodes_shortest_{}_{:.2f}.txt".format(len(nodes_shortest), shortest_distance)
        np.savetxt(fname_nodes_shortest, nodes_shortest, fmt="%i")

        # draw shortes calculated route
        draw(nodes_shortest, title="Calculated shortest route")

        # draw shortest given route
        draw(parser.get_coords_opt_tour(fname_opt_tour, fname_tsp, strip_node_num=False),
            title="Given shortest route")


    # calculate statistics
    mean_distance = np.mean(distances)
    sample_var_distance = np.std(distances, ddof=1)
    confidence_interval = (1.96*sample_var_distance / np.sqrt(len(solns)))

    print("Minimum distance given solution: {:.2f}".format(optimal_distance))
    print("Initial distance: {:.2f}".format(initial_distance))
    print("Average found distance: {:.2f} +- {:.2f}".format(mean_distance, confidence_interval))
    print("Minimum found distance: {:.2f}".format(shortest_distance))

    print("Elapsed time: {:.2f}s".format(time.time() -time_start))

    # draw specified node solution
    fname_tour = "results/nodes_shortest_51_438.48.txt"
    tour = parser.parse_file(fname_tour, strip_node_num=False, header_length=0)
    draw(tour, title="fname_tour")

    plt.show()

    return


if __name__ == '__main__':
    main()

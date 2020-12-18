import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import tqdm
import sys
import numba
import parser


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

    # return the minimal distance
    return tot_distance(nodes)


def convergence(nodes, markov_length_l, t0_l, n_runs, cooling_l):
    """Calculate data for convergence of solution depending on coolings schedule, t0 and
    markov_length

    Args:
        nodes : np array
             array of x and y coords of solution
        markov_length_l : list of ints
             lengths of the markov chain
        t0_l : list of floats
            initial temperatures
        n_runs : int
            number of runs of SA per parameter configuration
        cooling_l : list of strings
            cooling schedules

    Returns:
        data_df : panda DataFrame
             DataFrame with results for each parameter configuration
    """
    nodes_set = []
    for i in range(n_runs):
        np.random.shuffle(nodes)
        nodes_set.append(nodes)

    result_df = pd.DataFrame()

    data = []
    for cooling in cooling_l:
        for t0 in t0_l:
            for markov_length in markov_length_l:
                runs_distance = []

                for i in tqdm.tqdm(range(n_runs)):
                    nodes = nodes_set[i]
                    sa, dl = simulated_annealing(nodes, markov_length, t0, cooling)
                    runs_distance.append(dl)

                data.append([runs_distance, markov_length, t0, cooling])

    data_df = pd.DataFrame.from_records(data, columns=["Data", "Markov Length", "T0", 'Cooling'])
    data_df.to_csv("results/test1.csv")

    return data_df


def draw(nodes, title = "", ticks = True):
    """Draw the route given by notes

    Args:
        nodes : np array
             array of x and y coords of solution
        title : string
            title of plot
        ticks : boolean
            show ticks of axis of plot
    """
    plt.figure()
    if title:
        plt.title(title)
    x = [i[-2] for i in nodes]
    y = [i[-1] for i in nodes]
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, color='blue', zorder=1)
    plt.scatter(x, y, color='red', zorder=2)
    plt.xlabel("x [-]")
    plt.ylabel("y [-]")
    if not ticks:
        plt.xticks([])
        plt.yticks([])

    return


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
def simulated_annealing(nodes, markov_length, t0, cooling="LOG"):
    """Simulated annealing algorithm

    Args:
        nodes : np array
             array of x and y coords of a given initial solution
        markov_length : int
             length of the markov chain
        t0 : float
            initial temperature

    Returns:
        nodes : np array
             array of x and y coords of final solution
    """
    t = t0
    curr_iter = 0

    # keep track of best solution
    best_distance = tot_distance(nodes)
    best_nodes = nodes.copy()

    improv_iter = 0       # counter tracking the amount of iterations before an improvement is made
    improv_limit = 1000   # after this amount of iterations a better solution has to be found, else return
    max_iter = 1e5        # maximum amount of iterations before returning

    # if program takes too long: return
    # this terminating value max_iter can be set higher if you want your programs to run longer
    while curr_iter < max_iter:

        # inner loop, over the Markov chain
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
        if cooling == "LOG":
            t = t_log(curr_iter, t0)
        elif cooling == "QUAD":
            t = t_over_quadr(curr_iter, t0)
        elif cooling == "LINEAR":
            t = t_over_linear(curr_iter, t0)

        # check termination condition
        if tot_distance(nodes) < best_distance:
            best_distance = tot_distance(nodes)
            best_nodes = nodes.copy()
            improv_iter = 0
        else:
            improv_iter += 1
            if improv_iter >= improv_limit:
                return nodes, [0]

        curr_iter += 1

    print("While loop takes too long. Maybe adjust max_iter; returning current solution")

    return best_nodes, [0]


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
    """Cooling schedule with 1 over natural log

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
def t_over_linear(curr_iter,t0):
    """Cooling schedule with 1 over linear

    Args:
        curr_iter : int
            current iteration
        t0 : float
            initial temperature

    Returns:
        t : float
            temperature
    """
    return t0 / (1 + 0.01 * curr_iter)


@numba.njit
def t_over_quadr(curr_iter, t0):
    """Cooling schedule with 1 over quadratic

    Args:
        curr_iter : int
            current iteration
        t0 : float
            initial temperature

    Returns:
        t : float
            temperature
    """
    return t0 / (1 + 0.000002 * curr_iter**2)



@numba.njit
def two_opt(nodes):
    """2 opt swap

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

    # calculate convergence
    markov_length_l = [len(nodes) * i for i in range(1, 4)]
    t0_l = [4,5]
    n_runs = 5
    cooling_l = ["LOG", "LINEAR", "QUAD"]
    convergence(nodes, markov_length_l, t0_l, n_runs, cooling_l)

    # specify parameters for SA
    # t0, markov_multiplier, cooling = 11, 50, "LOG"
    t0, markov_multiplier, cooling = 40, 50, "LINEAR"
    # t0, markov_multiplier, cooling = 50, 40, "QUAD"
    # t0, markov_multiplier, cooling = 4, 1, "LOG"
    markov_length = len(nodes)*markov_multiplier
    n_runs = 5                        # number of runs of SA algorithm
    solns = []                       # list of final solutions per run

    print("Cooling, t0, markov_multiplier: ({}, {}, {})".format(cooling, t0, markov_multiplier))

    for i in tqdm.tqdm(range(n_runs)):

        # create random initial solution
        np.random.shuffle(nodes)
        initial_distance = tot_distance(nodes)

        # perform simulated annealing
        sa, dl = simulated_annealing(nodes, markov_length, t0, cooling=cooling)
        solns.append(sa)

    # save best solution
    distances = [tot_distance(soln) for soln in solns]
    shortest_distance = np.min(distances)
    nodes_shortest = solns[np.where(distances == shortest_distance)[0][0]]

    # node number has to be parsed to be able to save
    if len(solns[0][0]) == 3:
        fname_nodes_shortest = "results/nodes_shortest_{}_{:.2f}.txt".format(len(nodes_shortest),
            shortest_distance)
        np.savetxt(fname_nodes_shortest, nodes_shortest, fmt="%i")

        # draw shortes calculated and given route
        draw(nodes_shortest, title="Calculated shortest route")
        draw(parser.get_coords_opt_tour(fname_opt_tour, fname_tsp, strip_node_num=False),
            title="Given shortest route")

    # calculate statistics
    mean_distance = np.mean(distances)
    sample_var_distance = np.std(distances, ddof=1)
    confidence_interval = (1.96*sample_var_distance / np.sqrt(len(solns)))

    print("Minimum distance given solution: {:.2f}".format(optimal_distance))
    print("Initial distance: {:.2f}\n".format(initial_distance))

    print("Cooling, t0, markov_multiplier: ({}, {}, {})".format(cooling, t0, markov_multiplier))
    print("Average found distance: {:.2f} +- {:.2f}".format(mean_distance, confidence_interval))
    print("Minimum found distance: {:.2f}".format(shortest_distance))

    print("Elapsed time: {:.2f}s".format(time.time() -time_start))

    plt.show()

    return


if __name__ == '__main__':
    main()

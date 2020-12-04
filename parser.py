import numpy as np
import itertools


def parse_file(fname, strip_node_num=False):
    """Parse input file of TSP problem (tsp.txt) to array of node coordinates

    Args:
        fname : string
            file name of data file
        no_node_num : boolean
            removes node number from data, returns only the coordinate per node

    Returns:
        data : np array of lists ([#NODE, x, y] if strip_node_num=False, else [x, y])
            each list in the array represents a node
    """
    data = []
    with open(fname) as f:

        # read through file and strip first 6 header lines
        for line in itertools.islice(f, 6, None):

            # if EOF reached, stop reading input
            if line.strip("\n") == "EOF":
                break

            # strip "\n" and split at " "
            line = line.strip("\n").split(" ")

            # remove empty strings from line list.
            line = [number for number in line if number != ""]

            # remove node number
            if strip_node_num:
                line = line[1:]

            # convert all strings in line to float
            data.append([int(float(i)) for i in line])

    return np.array(data)


def parse_sol_file(fname):
    """Parse input file of TSP solution (opt.tour.txt) to array of nodes

    Args:
        fname : string
            file name of solution file

    Returns:
        data : np array
            array of node numbers
    """
    data = []
    with open(fname) as f:

        # read through file and strip first 6 header lines
        for line in itertools.islice(f, 5, None):

            # if EOF reached, stop reading input
            if line.strip("\n") == "EOF" or line.strip("\n") == "-1":
                break

            # strip "\n" and split at " "
            line = line.strip("\n")

            # convert all strings in line to float
            data.append(int(line))

    return np.array(data)


def get_coords_opt_tour(fname_opt_tour, fname_tsp):
    """Get corresponding coordinates for optimal tour file (opt.tour.txt)

    Args:
        fname_opt_tour : string
             file name of given optimal solution (opt.tour.txt)
        fname_tsp : string
             file name of given problem (tsp.txt)

    Returns:
        coords_opt_tour : np array of [x, y] coordinates
            each list in the array represents the coordinates of a node
    """
    nodes_tsp = parse_file(fname_tsp, strip_node_num=False)
    nodes_opt_tour = parse_sol_file(fname_opt_tour)

    coords_opt_tour = []
    for node_num in nodes_opt_tour:
        index = np.where(nodes_tsp[:,0] == node_num)[0][0]

        coords_opt_tour.append(nodes_tsp[index][1:])

    return np.array(coords_opt_tour)

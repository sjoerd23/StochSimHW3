# parser.py
# Helper file (not executable). Parses input files to be used in tsp.py

import numpy as np
import itertools


def parse_file(fname, strip_node_num=False, header_length=6):
    """Parse input file of TSP problem (tsp.txt) to array of node coordinates

    Args:
        fname : string
            file name of data file
        strip_node_num : boolean
            removes node number from data, returns only the coordinate per node
        header_length : int (default: 6)
            length of the header of the data file. Set to 0 if loading a generated solution

    Returns:
        data : np array of lists ([#NODE, x, y] if strip_node_num=False, else [x, y])
            each list in the array represents a node
    """
    data = []
    with open(fname) as f:

        # read through file and strip first header_length lines
        for line in itertools.islice(f, header_length, None):

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

        # read through file and strip first header_length lines
        if fname == "data/a280.opt.tour.txt":
            header_length = 4
        else:
            header_length = 5
        for line in itertools.islice(f, header_length, None):

            # if EOF reached, stop reading input
            if line.strip("\n") == "EOF" or line.strip("\n") == "-1":
                break

            # strip "\n" and split at " "
            line = line.strip("\n")

            # convert all strings in line to float
            data.append(int(line))

    return np.array(data)


def get_coords_opt_tour(fname_opt_tour, fname_tsp, strip_node_num=False):
    """Get corresponding coordinates for optimal tour file (opt.tour.txt)

    Args:
        fname_opt_tour : string
             file name of given optimal solution (opt.tour.txt)
        fname_tsp : string
             file name of given problem (tsp.txt)
        strip_node_num : boolean
            removes node number from data, returns only the coordinate per node

    Returns:
        coords_opt_tour : np array of [x, y] coordinates
            each list in the array represents the coordinates of a node
    """
    nodes_tsp = parse_file(fname_tsp, strip_node_num=False)
    nodes_opt_tour = parse_sol_file(fname_opt_tour)

    coords_opt_tour = []
    for node_num in nodes_opt_tour:
        index = np.where(nodes_tsp[:, 0] == node_num)[0][0]

        if strip_node_num:
            coords_opt_tour.append(nodes_tsp[index][1:])
        else:
            coords_opt_tour.append(nodes_tsp[index][:])

    return np.array(coords_opt_tour)

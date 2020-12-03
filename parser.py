import numpy as np
import itertools


def parse_file(fname, strip_node_num=False):
    """Parse input file of TSP problem to array of node coordinates

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

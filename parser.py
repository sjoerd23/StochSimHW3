import numpy as np
import itertools


def parse_file(fname):
    """Parse input file of TSP problem to array of node coordinates

    Args:
        fname : string
            file name of data file

    Returns:
        data : np array of lists
            each list in the array represents a node and its coordinate

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

            # remove empty strings from line list
            line = [number for number in line if number != ""]

            # convert all strings in line to float
            data.append([int(float(i)) for i in line])

    return np.array(data)

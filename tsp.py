import numpy as np
import matplotlib.pyplot as plt
import parser


def distance(node):
    if len(node) == 2:
        return np.sqrt(node[0]**2 + node[1]**2)
    elif len(node) == 3:
        return np.sqrt(node[1]**2 + node[2]**2)


def main():

    fname = "data/eil51.tsp.txt"
    # fname = "data/a280.tsp.txt"
    # fname = "data/pcb442.tsp.txt"

    # parse tsp.txt file to data
    data = parser.parse_file(fname, strip_node_num=False)

    print([distance(node) for node in data])


    return


if __name__ == '__main__':
    main()

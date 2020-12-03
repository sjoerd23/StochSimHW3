import numpy as np
import matplotlib.pyplot as plt
import parser


def main():

    fname = "data/eil51.tsp.txt"
    fname = "data/a280.tsp.txt"
    fname = "data/pcb442.tsp.txt"

    # parse tsp.txt file to data
    data = parser.parse_file(fname, strip_node_num=True)
    print(data)

    return


if __name__ == '__main__':
    main()

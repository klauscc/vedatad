# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================

import multiprocessing
import numpy as np


def print_matrix(filename):
    matrix = np.memmap(filename, dtype="int16", mode="r", shape=(100, 100))
    print(matrix)


def main():
    filename = "test.dat"
    matrix = np.memmap(filename, dtype="int16", mode="w+", shape=(100, 100))
    matrix[0] = -2
    matrix.flush()
    print(matrix)
    p = multiprocessing.Process(target=print_matrix, args=(filename,))
    p.start()
    p.join()
    print(matrix)


if __name__ == "__main__":
    main()

from tools.matrix_util import Matrix
from tools.random_util import random_normal_matrix
from python import Python
from random import seed
from math import sqrt
from tensor import Tensor
from utils.index import Index

def plot_matrix_row(tensor: Tensor[DType.float32], row: Int, length: Int):
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    row_array = np.zeros(length, np.float32)

    for col in range(length):
        row_array.itemset(col, tensor[row, col])

    plt.hist(row_array)
    plt.savefig("row_hist.png")
    plt.show()

def plot_matrix_col(tensor: Tensor[DType.float32], col: Int, length: Int):
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    row_array = np.zeros(length, np.float32)

    for row in range(length):
        row_array.itemset(row, tensor[row, col])

    plt.hist(row_array)
    plt.savefig("col_hist.png")
    plt.show()

fn main():

    let M = Matrix(1000, 1000)
    random_normal_matrix(1.0, 10.0, M)

    print('sampled means are')
    print(M.row_mean(100))
    print(M.col_mean(101))

    print('sampled stds are')
    print(M.row_std(100))
    print(M.col_std(101))

    # copy Matrix to a Tensor
    var T = Tensor[DType.float32](M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            T[Index(i, j)] = M[i, j]

    try:
        _ = plot_matrix_col(T, 10, 1000)
    except e:
        print("failed to show plot:", e.value)

    try:
        _ = plot_matrix_row(T, 10, 1000)
    except e:
        print("failed to show plot:", e.value)
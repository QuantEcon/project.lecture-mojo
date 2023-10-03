from tools.matrix_util import Matrix, matmul, transpose
from time import now
from benchmark import Benchmark
from runtime.llcl import Runtime

fn small_matmul():
    let A = Matrix(2, 2)

    print('matrix A:')
    A.print()

    let AT = Matrix(A.rows, A.cols)
    
    transpose(A, AT)

    print('matrix A.T:')
    AT.print()

    var B = Matrix(A.rows, AT.cols)
    B.zero()

    matmul(B, A, AT)

    print('matrix A @ A.T:')
    B.print()

fn large_matmul():
    let size = 3000
    
    let A = Matrix(size, size)
    let AT = Matrix(A.rows, A.cols)
    
    transpose(A, AT)

    var B = Matrix(A.rows, AT.cols)
    B.zero()

    @always_inline
    @parameter
    fn test_fn():
        matmul(B, A, AT)

    var secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    print("Completed matmul in ", secs * 1_000, "ms /", secs, "s")

    let gflops = ((2*size**3)/secs) / 1e9
    print(gflops, "GFLOP/s")

fn main():
    small_matmul()
    large_matmul()

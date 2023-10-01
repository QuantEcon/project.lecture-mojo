from tools import Matrix, matmul, transpose
from time import now

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

    let eval_begin: Float64 = now()
    matmul(B, A, AT)
    let eval_end: Float64 = now()

    let execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed naive matmul in ", execution_time, "ms")

    let gflops = ((2*size**3)/execution_time) / 1e9
    print(gflops, "GFLOP/s")

fn main():
    small_matmul()
    large_matmul()

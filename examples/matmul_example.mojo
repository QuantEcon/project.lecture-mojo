from benchmark import Benchmark
from runtime.llcl import Runtime
from memory import memset_zero
from random import rand, seed
from algorithm import parallelize, vectorize
from math import exp, sqrt
from time import now

# Based on Matrix Multiplication Example in Mojo: https://docs.modular.com/mojo/notebooks/Matmul.html
alias nelts = simdwidthof[DType.float32]()

# Define the Matrix struct
struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int, seed_int: Int = 0):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        seed(seed_int)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)
    fn print(self):
        print_no_newline('[')
        for m in range(self.rows):
            print_no_newline('[')
            for n in range(self.cols):
                if n != 0:
                    print_no_newline(', ')
                print_no_newline(self[m, n])
            if m != self.rows - 1:
                print_no_newline('],\n')
            else:
                print_no_newline(']')
        print_no_newline(']')
        print('\n')

fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]
                
fn matmul(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)
        
    parallelize[calc_row](C.rows)

# # Faster matrix multiplication algorithm
# fn matmul(self, other: Matrix) -> Matrix:
#     assert(self.cols == other.rows, "Matrix dimensions mismatch")
#     var result = Matrix(self.rows, other.cols)

#     for i in 0..self.rows:
#         for j in 0..other.cols:
#             var sum = simd<0.0, 0.0, 0.0, 0.0>()
#             for k in 0..self.cols:
#                 sum += self.load[nelts](i, k) * other.load[nelts](k, j)
#             result.store[nelts](i, j, sum)
    
#     return result

fn transpose(A: Matrix, AT: Matrix):
    for m in range(A.rows):
        for n in range(A.cols):
            AT[n, m] = A[m, n]


fn large_matmul():
    let size = 5000
    
    let A = Matrix(size, size)
    let AT = Matrix(A.rows, A.cols)
    
    transpose(A, AT)

    var B = Matrix(A.rows, AT.cols)
    B.zero()

    @always_inline
    @parameter
    fn test_fn():
        matmul(B, A, AT)

    # var secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    # print("Completed matmul in ", secs * 1_000, "ms /", secs, "s")

    # let gflops = ((2*size**3)/secs) / 1e9
    # print(gflops, "GFLOP/s")

    let eval_begin: Float64 = now()
    test_fn()
    let eval_end: Float64 = now()

    let execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed matmul in ", execution_time, "ms")
    let secs = execution_time / 1000
    let gflops = ((2*size**3)/secs) / 1e9
    print(gflops, "GFLOP/s")

fn main():
    large_matmul()
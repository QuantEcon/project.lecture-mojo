# Based on Matrix Multiplication Example in Mojo: https://docs.modular.com/mojo/notebooks/Matmul.html

from memory import memset_zero
from random import rand, seed
from algorithm import parallelize, vectorize

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

fn matmul(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)
        
    parallelize[calc_row](C.rows)
        
fn transpose(A: Matrix, AT: Matrix):
    for m in range(A.rows):
        for n in range(A.cols):
            AT[n, m] = A[m, n]
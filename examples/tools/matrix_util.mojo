from memory import memset_zero
from random import rand, seed
from algorithm import parallelize, vectorize
from math import exp, sqrt
from time import now
from random import rand, seed, random_float64
from algorithm import parallelize
from math import sqrt, log, cos
from math.limit import inf, neginf

alias PI = 3.141592653589793

# Based on Matrix Multiplication Example in Mojo: https://docs.modular.com/mojo/notebooks/Matmul.html
alias nelts = simdwidthof[DType.float64]()

# Define the Matrix struct
struct Matrix:
    var data: DTypePointer[DType.float64]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int, seed_int: Int = 0):
        self.data = DTypePointer[DType.float64].alloc(rows * cols)
        seed(seed_int)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float64:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float64):
        return self.store[1](y, x, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float64, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float64, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    @always_inline
    fn row_sum(self, row: Int) -> Float64:
        var sum: Float64 = 0.0
        for n in range(self.cols):
            sum += self[row, n]
        return sum

    @always_inline
    fn col_sum(self, col: Int) -> Float64:
        var sum: Float64 = 0.0
        for m in range(self.rows):
            sum += self[m, col]
        return sum

    @always_inline
    fn row_mean(self, row: Int) -> Float64:
        return self.row_sum(row) / self.cols

    @always_inline
    fn col_mean(self, col: Int) -> Float64:
        return self.col_sum(col) / self.rows

    @always_inline
    fn row_std(self, row: Int) -> Float64:
        let mean: Float64 = self.row_mean(row)
        var sum: Float64 = 0.0
        for n in range(self.cols):
            sum += (self[row, n] - mean) * (self[row, n] - mean)
        return sqrt(sum / self.cols)
    
    @always_inline
    fn col_std(self, col: Int) -> Float64:
        let mean: Float64 = self.col_mean(col)
        var sum: Float64 = 0.0
        for m in range(self.rows):
            sum += (self[m, col] - mean) * (self[m, col] - mean)
        return sqrt(sum / self.rows)

    @always_inline            
    fn max(self) -> Float64:
        var max_val: Float64 = neginf[DType.float64]()
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j] > max_val:
                    max_val = self[i, j]
        return max_val

    @always_inline            
    fn min(self) -> Float64:
        var min_val: Float64 = inf[DType.float64]()
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j] < min_val:
                    min_val = self[i, j]
        return min_val

    fn fill_Matrix(self, value:Float64):
        for i in range(self.rows):
            for j in range(self.cols):
                self[i, j] = value

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
        
fn transpose(A: Matrix, AT: Matrix):
    for m in range(A.rows):
        for n in range(A.cols):
            AT[n, m] = A[m, n]

fn scalar_mul(A: Matrix, m: Float64):
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] = m*A[i, j]

fn scalar_add(A: Matrix, a: Float64):
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] += a

fn scalar_exp(A: Matrix):
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] = exp(A[i, j])

fn matrix_add(A: Matrix, B: Matrix):
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] += B[i, j]

fn random_normal(mu: Float64, sigma: Float64, M: SIMD[DType.float64, 1]) -> SIMD[DType.float64, 1]:

    let u1 = random_float64().cast[DType.float64]()
    let u2 = random_float64().cast[DType.float64]()

    let z0 = sqrt(-2 * log(u1)) * cos[DType.float64, 1](2 * PI * u2)

    return sigma*z0 + mu
    

fn random_normal_matrix(mu: Float64, sigma: Float64, M: Matrix):
    @parameter
    fn calc_row(i: Int):
        for j in range(M.cols):
            M[i, j] = random_normal(mu, sigma, M[i, j])
    parallelize[calc_row](M.rows)


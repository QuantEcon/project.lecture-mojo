from random import rand, seed, random_float64
from tools.matrix_util import Matrix
from algorithm import parallelize
from math import sqrt, log, cos
from python import Python

alias PI = 3.141592653589793


fn random_normal(mu: Float32, sigma: Float32, M: SIMD[DType.float32, 1]) -> SIMD[DType.float32, 1]:

    let u1 = random_float64().cast[DType.float32]()
    let u2 = random_float64().cast[DType.float32]()

    let z0 = sqrt(-2 * log(u1)) * cos[DType.float32, 1](2 * PI * u2)

    return sigma*z0 + mu
    

fn random_normal_matrix(mu: Float32, sigma: Float32, M: Matrix):
    @parameter
    fn calc_row(i: Int):
        for j in range(M.cols):
            M[i, j] = random_normal(mu, sigma, M[i, j])
    parallelize[calc_row](M.rows)



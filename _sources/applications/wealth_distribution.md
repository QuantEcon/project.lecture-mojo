---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Mojo
  language: mojo
  name: mojo-jupyter-kernel
---

# Wealth Distribution Dynamics

Let's try to re-write the [Wealth Distribution Dynamics](https://jax.quantecon.org/wealth_dynamics.html)
in mojo and compare the results with JAX.

Let's start with the following imports

```{code-cell}
from memory import memset_zero, memcpy
from random import rand, random_float64
from math import abs, exp, sqrt, log, cos
from time import now

# Set the constant value of PI
alias PI = 3.141592653589793
```

We define the random normal variate generator functions
that will be used later in the example.

```{code-cell}
fn random_normal(mu: Float32, sigma: Float32, M: SIMD[DType.float32, 1]) -> SIMD[DType.float32, 1]:

    let u1 = random_float64().cast[DType.float32]()
    let u2 = random_float64().cast[DType.float32]()

    let z0 = sqrt(-2 * log(u1)) * cos[DType.float32, 1](2 * PI * u2)

    return sigma*z0 + mu


fn random_normal_matrix(mu: Float32, sigma: Float32, M: Matrix):
    for i in range(M.rows):
        for j in range(M.cols):
            M[i, j] = random_normal(mu, sigma, M[i, j])
```

We need some matrix operations like multiplication, addition,
and exponential of the whole matrix by some scalar number.

```{code-cell}
fn scalar_mul(A: Matrix, m: Float32):
    # Multiplies each element of A by m
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] = m*A[i, j]

fn scalar_add(A: Matrix, a: Float32):
    # Adds each element of A by a
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] += a

fn scalar_exp(A: Matrix):
    # Exponential of each element of A
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] = exp(A[i, j])

fn matrix_add(A: Matrix, B: Matrix):
    # Adds two Matrix A and B. Result is stored in A
    for i in range(A.rows):
        for j in range(A.cols):
            A[i, j] += B[i, j]
```

## A Model of Wealth Dynamics

The model we will study is

$$
    w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
$$

where

- $w_t$ is wealth at time $t$ for a given household,
- $r_t$ is the rate of return of financial assets,
- $y_t$ is current non-financial (e.g., labor) income and
- $s(w_t)$ is current wealth net of consumption


The following function updates one period with the given
current wealth and persistent state.

```{code-cell}
fn update_states_mojo(w: Matrix, z: Matrix, a: Float32, b: Float32, sigma_z: Float32,
                    c_y: Float32, sigma_y: Float32, mu_y: Float32,
                    c_r: Float32, sigma_r: Float32, mu_r: Float32, s_0: Float32,
                    w_hat: Float32):

    scalar_mul(z, a)
    scalar_add(z, b)
    let rand_matrix1: Matrix = Matrix(z.rows, z.cols)
    random_normal_matrix(0, 1, rand_matrix1)
    scalar_mul(rand_matrix1, sigma_z)
    matrix_add(z, rand_matrix1)

    var zp: Matrix = Matrix(z)

    scalar_exp(zp)
    scalar_mul(zp, c_y)
    let rand_matrix2: Matrix = Matrix(z.rows, z.cols)
    random_normal_matrix(0, 1, rand_matrix2)
    scalar_mul(rand_matrix2, sigma_y)
    scalar_add(rand_matrix2, mu_y)
    scalar_exp(rand_matrix2)
    matrix_add(zp, rand_matrix2)

    let mat_y: Matrix =  Matrix(zp)

    zp = Matrix(z)
    scalar_exp(zp)
    scalar_mul(zp, c_r)
    let rand_matrix3: Matrix = Matrix(z.rows, z.cols)
    random_normal_matrix(0, 1, rand_matrix3)
    scalar_mul(rand_matrix3, sigma_r)
    scalar_add(rand_matrix3, mu_r)
    scalar_exp(rand_matrix3)
    matrix_add(zp, rand_matrix3)

    let R: Matrix =  Matrix(zp)
    for i in range(w.rows):
        for j in range(w.cols):
            if w[i, j] >= w_hat:
                mat_y[i, j] += R[i, j] * s_0 * w[i, j]
            w[i, j] = mat_y[i, j]
```

```{code-cell}
fn wealth_time_series(result: Matrix, n: Int, w_0: Float32, a: Float32, b: Float32, sigma_z: Float32,
                    c_y: Float32, sigma_y: Float32, mu_y: Float32,
                    c_r: Float32, sigma_r: Float32, mu_r: Float32, s_0: Float32,
                    w_hat:Float32, z_mean: Float32, z_var: Float32):

    let z: Matrix = Matrix(1, 1)
    random_normal_matrix(0, 1, z)
    scalar_mul(z, sqrt(z_var))
    scalar_add(z, z_mean)
    let w: Matrix = Matrix(1, 1)
    w[0, 0] = w_0
    result[0, 0] = w_0
    for i in range(n-1):
        update_states_mojo(w, z, a, b, sigma_z,
                    c_y, sigma_y, mu_y,
                    c_r, sigma_r, mu_r, s_0, w_hat)
        result[i+1, 0] = w[0, 0]
```

Let's combine all these functions and simulate the time series.

```{code-cell}
fn execute_wealth_time_series():
    let w_hat: Float32 = 1.0
    let s_0: Float32 = 0.75
    let c_y: Float32 = 1.0
    let mu_y: Float32 = 1.0
    let sigma_y: Float32 = 0.2
    let c_r: Float32 = 0.05
    let mu_r: Float32 = 0.1
    let sigma_r: Float32 = 0.5
    let a: Float32 = 0.5
    let b: Float32 = 0.0
    let sigma_z: Float32 = 0.1
    let z_mean: Float32 = b / (1 - a)
    let z_var: Float32 = sigma_z**2 / (1 - a**2)
    let exp_z_mean: Float32 = exp(z_mean + z_var / 2)
    let y_mean: Float32 = c_y * exp_z_mean + exp(mu_y + sigma_y**2 / 2)
    let ts_length: Int = 200
    let result: Matrix = Matrix(ts_length, 1)
    let eval_begin: Float64 = now()
    wealth_time_series(result, ts_length, y_mean, a, b, sigma_z,
                    c_y, sigma_y, mu_y,
                    c_r, sigma_r, mu_r, s_0, w_hat, z_mean, z_var)
    let eval_end: Float64 = now()

    let execution_time = Float64((eval_end - eval_begin)) / 1e6
    
    # Uncomment the following lines to check the result array.
    # print("Result:")
    # for i in range(ts_length):
    #     print_no_newline(result[i, 0])
    #     if i != ts_length - 1:
    #         print_no_newline(', ')
    # print('')

    print("Completed wealth distribution in ", execution_time, "ms")

execute_wealth_time_series()
```

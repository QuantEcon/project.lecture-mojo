---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Mojo
  language: mojo
  name: mojo-jupyter-kernel
---

# Mojo Tutorial

This notebook walks through basic comparison of python and [mojo](https://docs.modular.com/mojo/)
and it's application in quantitative economics.

The notebook is built on top of
[mojo example](https://github.com/modularml/mojo/blob/main/examples/notebooks/HelloMojo.ipynb).


Mojo is a powerful programming language that's primarily designed for
high-performance systems programming, so it has a lot in common with
other systems languages like Rust and C++. Yet, Mojo is also designed
to become a superset of Python, so a lot of language features
and concepts you might know from Python translate nicely to Mojo.

```{code-cell}
print("Hello Mojo!", "Hello QuantEcon!")
```

Mojo preserves Python's dynamic features and language syntax, and
it even allows you to import and run code from Python packages.
However, it's important to know that Mojo is an entirely new
language, not just a new implementation of Python with syntax sugar.

Mojo takes the Python language to a whole new level, with
systems programming features, strong type-checking, memory safety,
next-generation compiler technologies, and more. Yet, it's still
designed to be a simple language that's useful for general-purpose programming.

First and foremost, Mojo is a compiled language and a lot of its
performance and memory-safety features are derived from that fact.
Mojo code can be ahead-of-time (AOT) or just-in-time (JIT) compiled.

## Variables

You can declare variables with `var` to create a mutable value,
or with `let` to create an immutable value.

Let's observe the difference in using them.

```{code-cell}
fn main():
    var x: Int = 10
    x += 1
    print(x)

main()
```

This works fine as expected. Now, let's replace `var` with `let`
and notice the error.

```{code-cell}
fn main():
    let x: Int = 10
    x += 1
    print(x)

main()
```

That's because `let` makes the value immutable, so you can't increment it.

## Function


Mojo functions can be declared with either `fn` or `def` (as in Python).
The `fn` declaration enforces
strongly-typed and memory-safe behaviors, while `def` provides
Python-style dynamic behaviors.

```{code-cell}
fn some_fun():
    let x: Int = 1
    let y = 2
    print(x + y)

some_fun()
```

Notice that the `x` variable has an explicit `Int` type specification.
Declaring the type is not required for variables in `fn`,
but it is desirable sometimes. If you omit it, Mojo infers the type.

Although types aren't required for variables declared in the function body,
they are required for arguments and return values for an `fn` function.

For example, here's how to declare `Int` as the type for function
arguments and the return value:

```{code-cell}
fn add_two_ints(x: Int, y: Int) -> Int:
    return x + y

z = add_two_ints(1, 2)
print(z)
```

If you want the arguments to be *mutable*, you need to declare each
argument convention as `inout`. This means that changes made to
the arguments *inside* the function are visible outside the function.

```{code-cell}
fn add_inout(inout x: Int, inout y: Int) -> Int:
    x += 1
    y += 1
    return x + y

var a = 1
var b = 2
print("Values of a and b before calling the function:", a, ",", b)

c = add_inout(a, b)

print("Values of a and b after calling the function:", a, ",", b)
print("Value of c:", c)
```

Notice how the values of `a` and `b` are changed due to
mutable definition using `inout`.

## Structures

We can build high-level abstractions for types (or "objects") in a struct.
A struct in Mojo is similar to a class in Python: they both support methods,
fields, operator overloading, decorators for metaprogramming, etc.
However, Mojo structs are completely static—they are bound at compile-time,
so they do not allow dynamic dispatch or any runtime
changes to the structure.

For example, here's a basic struct:

```{code-cell}
struct MyPair:
    var first: Int
    var second: Int

    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second
    
    fn dump(self):
        print(self.first, self.second)
```

```{code-cell}
let my_pair = MyPair(2, 4)
my_pair.dump()
```

## Application 1: Shortest Paths

Let's try to re-write the [Shortest Paths lecture](https://jax.quantecon.org/short_path.html)
in mojo and compare the results with JAX.

Let's start with the following imports

```{code-cell}
from utils.list import Dim
from memory import memset_zero
from random import rand
from math import min, abs
from time import now
```

Now, define the Matrix struct that allows storing and easy computation.

```{code-cell}
# Define the Matrix struct

struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols
    
    fn __init__(inout self, other: Matrix):
        self.rows = other.rows
        self.cols = other.cols
        self.data = DTypePointer[DType.float32].alloc(self.rows * self.cols)
        memcpy[DType.float32](self.data, other.data, self.rows * self.cols)

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
```

## Solving for Minimum Cost-to-Go

Let $J(v)$ denote the minimum cost-to-go from node $v$,
understood as the total cost from $v$ if we take the best route.

Let's look at an algorithm for computing $J$ and then think about how to
implement it.

### The Algorithm

The standard algorithm for finding $J$ is to start an initial guess
and then iterate.

This is a standard approach to solving nonlinear equations, often called
the method of **successive approximations**.

Our initial guess will be

$$
J_0(v) = 0 \text{ for all } v
$$

Now

1. Set $n = 0$
2. Set $J_{n+1} (v) = \min_{w \in F_v} \{ c(v, w) + J_n(w) \}$ for all $v$
3. If $J_{n+1}$ and $J_n$ are not equal then increment $n$, go to 2

This sequence converges to $J$.

Let's start by defining the **distance matrix** $Q$.

Let's define the function that fills all the elements of matrix `Q` with `inf`.

```{code-cell}
# Fill all the elements of Matrix with inf

fn fill_Q_Matrix(Q: Matrix):
    let inf: Float32 = 100000000.00
    for i in range(Q.rows):
        for j in range(Q.cols):
            Q[i, j] = inf
```

The following utility function computes the result equivalent
to `np.sum(Q + J, axis=1)`.

Here, we use `J[i]` vector element as `J[0, i]` as `J` is an
object of `Matrix` struct which requires 2-D instantiation.

```{code-cell}
# Returns equivalent to np.sum(Q + J, axis=1)

fn add_and_min_axis_1(Q: Matrix, J: Matrix, new_J: Matrix):
    let inf: Float32 = 100000000.00
    for i in range(Q.rows):
        var min_value: Float32 = inf
        for j in range(Q.cols):
            min_value = min(Q[i, j] + J[0, j], min_value)
        new_J[0, i] = min_value
```

To check where two matrices are almost equal, we define
the following function equivalent to `np.allclose`.

```{code-cell}
# Checks whether both the matrices are almost equal

fn check_close(J: Matrix, new_J: Matrix) -> Bool:
    let inf: Float32 = 100000000.00
    for i in range(J.cols):
        if abs(new_J[0, i] - J[0, i]) > 1e-5:
            return False
    return True
```

```{code-cell}
# Compute the shortest path

fn shortest_paths(Q: Matrix, J: Matrix):
    let max_iter: Int = 500
    var i: Int = 0
    while i < max_iter:
        let new_J: Matrix = Matrix(1, 7)
        add_and_min_axis_1(Q, J, new_J)
        if check_close(J, new_J):
            break
        for j in range(7):
            J[0, j] = new_J[0, j]
        i += 1
```

Now, let's wire all these functions and run the shortest paths
on some example matrix.

```{code-cell}
fn execute_shortest_paths():
    let Q: Matrix = Matrix(7, 7)
    fill_Q_Matrix(Q)
    Q[0, 1] = 1.0
    Q[0, 2] = 5.0
    Q[0, 3] = 3.0
    Q[1, 3] = 9.0
    Q[1, 4] = 6.0
    Q[2, 5] = 2.0
    Q[3, 5] = 4.0
    Q[3, 6] = 8.0
    Q[4, 6] = 4.0
    Q[5, 6] = 1.0
    Q[6, 6] = 0.0

    var J: Matrix = Matrix(1, 7)
    J.zero()

    let eval_begin: Float64 = now()
    shortest_paths(Q, J)
    let eval_end: Float64 = now()

    let execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed shortest paths in", execution_time, "ms")
    print("The cost-to-go value is:")
    for i in range(7):
        print(J[0, i])

execute_shortest_paths()
```

## Application 2: Wealth Distribution Dynamics

Let's try to re-write the [Wealth Distribution Dynamics](https://jax.quantecon.org/wealth_dynamics.html)
in mojo and compare the results with JAX.

Let's start with the following imports

```{code-cell}
from memory import memset_zero, memcpy
from random import rand, random_float64
from math import abs, exp, sqrt, log, cos
from time import now


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

### A Model of Wealth Dynamics

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

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


# Shortest Paths

Let's try to re-write the [Shortest Paths lecture](https://jax.quantecon.org/short_path.html)
in mojo and compare the results with JAX.

Let's start with the following imports

```{code-cell} mojo
from utils.list import Dim
from memory import memset_zero
from random import rand
from math import min, abs
from time import now
```

Now, define the Matrix struct that allows storing and easy computation.

```{code-cell} mojo
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

```{code-cell} mojo
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

```{code-cell} mojo
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

```{code-cell} mojo
# Checks whether both the matrices are almost equal

fn check_close(J: Matrix, new_J: Matrix) -> Bool:
    let inf: Float32 = 100000000.00
    for i in range(J.cols):
        if abs(new_J[0, i] - J[0, i]) > 1e-5:
            return False
    return True
```

```{code-cell} mojo
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

```{code-cell} mojo
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
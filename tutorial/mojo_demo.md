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

This notebook walks through basic comparison of python and [mojo](https://docs.modular.com/mojo/) and it's application in quantitative economics.

The notebook is built on top of [mojo example](https://github.com/modularml/mojo/blob/main/examples/notebooks/HelloMojo.ipynb).



Mojo is a powerful programming language that's primarily designed for high-performance systems programming, so it has a lot in common with other systems languages like Rust and C++. Yet, Mojo is also designed to become a superset of Python, so a lot of language features and concepts you might know from Python translate nicely to Mojo.

```{code-cell}
print("Hello Mojo!", "Hello QuantEcon!")
```

Mojo preserves Python's dynamic features and language syntax, and it even allows you to import and run code from Python packages. However, it's important to know that Mojo is an entirely new language, not just a new implementation of Python with syntax sugar.

Mojo takes the Python language to a whole new level, with systems programming features, strong type-checking, memory safety, next-generation compiler technologies, and more. Yet, it's still designed to be a simple language that's useful for general-purpose programming.



First and foremost, Mojo is a compiled language and a lot of its performance and memory-safety features are derived from that fact. Mojo code can be ahead-of-time (AOT) or just-in-time (JIT) compiled.



## Variables



You can declare variables with `var` to create a mutable value, or with `let` to create an immutable value.

Let's observe the difference in using them.

```{code-cell}
fn main():
    var x: Int = 10
    x += 1
    print(x)

main()
```

This works fine as expected. Now, let's replace `var` with `let` and notice the error.

```{code-cell}
---
tags: [raises-exception]
---

fn main():
    let x: Int = 10
    x += 1
    print(x)

main()
```

That's because `let` makes the value immutable, so you can't increment it.



## Function



Mojo functions can be declared with either `fn` or `def` (as in Python). The `fn` declaration enforces
strongly-typed and memory-safe behaviors, while `def` provides Python-style dynamic behaviors.

```{code-cell}
fn some_fun():
    let x: Int = 1
    let y = 2
    print(x + y)

some_fun()
```

Notice that the `x` variable has an explicit `Int` type specification. Declaring the type is not required for variables in `fn`, but it is desirable sometimes. If you omit it, Mojo infers the type.

Although types aren't required for variables declared in the function body, they are required for arguments and return values for an `fn` function.

For example, here's how to declare `Int` as the type for function arguments and the return value:

```{code-cell}
fn add_two_ints(x: Int, y: Int) -> Int:
    return x + y

z = add_two_ints(1, 2)
print(z)
```

If you want the arguments to be *mutable*, you need to declare each argument convention as `inout`. This means that changes made to the arguments *inside* the function are visible outside the function.

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

Notice how the values of `a` and `b` are changed due to mutable definition using `inout`.



## Structures



You can build high-level abstractions for types (or "objects") in a struct. A struct in Mojo is similar to a class in Python: they both support methods, fields, operator overloading, decorators for metaprogramming, etc. However, Mojo structs are completely static—they are bound at compile-time, so they do not allow dynamic dispatch or any runtime changes to the structure.

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

Let's try to re-write the [Shortest Paths lecture](https://jax.quantecon.org/short_path.html) in mojo and compare the results with JAX.

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

Let's define the function that fills all the elements of matrix `Q` with `inf`.

```{code-cell}
# Fill all the elements of Matrix with inf
fn fill_Q_Matrix(Q: Matrix):
    let inf: Float32 = 100000000.00
    for i in range(Q.rows):
        for j in range(Q.cols):
            Q[i, j] = inf
```

The following utility function computes the result equivalent to `np.sum(Q + J, axis=1)`.

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

To check where two matrices are almost equal, we define the following function equivalent to `np.allclose`.

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

Now, let's wire all these functions and run the shortest paths on some example matrix.

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
    print("Completed naive shortest paths in ", execution_time, "ms")
    print("The cost-to-go value is:")
    for i in range(7):
        print(J[0, i])

execute_shortest_paths()
```

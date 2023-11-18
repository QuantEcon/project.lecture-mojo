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

```{code-cell} mojo
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

```{code-cell} mojo
fn main():
    var x: Int = 10
    x += 1
    print(x)

main()
```

This works fine as expected. Now, let's replace `var` with `let`
and notice the error.

```{code-cell} mojo
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


Mojo functions can be declared with either `fn` or `def` (as in Python).
The `fn` declaration enforces
strongly-typed and memory-safe behaviors, while `def` provides
Python-style dynamic behaviors.

```{code-cell} mojo
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

```{code-cell} mojo
fn add_two_ints(x: Int, y: Int) -> Int:
    return x + y

z = add_two_ints(1, 2)
print(z)
```

If you want the arguments to be *mutable*, you need to declare each
argument convention as `inout`. This means that changes made to
the arguments *inside* the function are visible outside the function.

```{code-cell} mojo
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

```{code-cell} mojo
struct MyPair:
    var first: Int
    var second: Int

    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second
    
    fn dump(self):
        print(self.first, self.second)
```

```{code-cell} mojo
let my_pair = MyPair(2, 4)
my_pair.dump()
```
from math import min, abs
from time import now
from tools import Matrix

# Fill all the elements of Matrix with inf
fn fill_Q_Matrix(Q: Matrix):
    let inf: Float32 = 100000000.00
    for i in range(Q.rows):
        for j in range(Q.cols):
            Q[i, j] = inf

# Returns equivalent to np.sum(Q + J, axis=1)
fn add_and_min_axis_1(Q: Matrix, J: Matrix, new_J: Matrix):
    let inf: Float32 = 100000000.00
    for i in range(Q.rows):
        var min_value: Float32 = inf
        for j in range(Q.cols):
            min_value = min(Q[i, j] + J[0, j], min_value)
        new_J[0, i] = min_value

# Checks whether both the matrices are almost equal
fn check_close(J: Matrix, new_J: Matrix) -> Bool:
    let inf: Float32 = 100000000.00
    for i in range(J.cols):
        if abs(new_J[0, i] - J[0, i]) > 1e-5:
            return False
    return True

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

fn main():
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

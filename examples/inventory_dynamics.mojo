from math import min, max, abs, exp
from time import now
from random import randn_float64
from tools.matrix_util import Matrix, random_normal_matrix, scalar_exp, scalar_add, scalar_mul, random_normal
from algorithm import parallelize, vectorize

struct Firm:
    var s: Int
    var S: Int
    var mu: Float64
    var sigma: Float64

    fn __init__(inout self, 
                s: Int, 
                S: Int,
                mu: Float64,
                sigma: Float64):
        self.s = s
        self.S = S
        self.mu = mu
        self.sigma = sigma


fn update_stock(X: Matrix, s: Int, S: Int, D: Matrix, n_restock: Matrix):
        for i in range(X.rows):
                let X_val = X[i, 1]
                if X[i, 1] <= s:
                        X[i, 1] = max(S - D.min(), 0)
                        n_restock[i, 1] = n_restock[i, 1] + 1
                else:
                        X[i, 1] = max(X_val - D.min(), 0)

fn count_stock(n_restock: Matrix) -> Float64:
        var count: Int = 0
        for i in range(n_restock.rows):
                if n_restock[i, 1] > 1:
                        count += 1
        return count / (n_restock.rows + 1)

fn shift_firms_forward(x_init: Float64, 
                       firm: Firm, 
                       num_firms: Int, 
                       sim_length: Int):

        let s: Int = firm.s
        let S: Int = firm.S
        let mu: Float64 = firm.mu
        let sigma: Float64 = firm.sigma

        let X: Matrix = Matrix(num_firms, 1)
        X.fill_Matrix(x_init)

        var n_restock: Matrix = Matrix(num_firms, 1)
        n_restock.zero()


        for i in range(sim_length):
                let Z: Matrix = Matrix(num_firms, 1) 
                random_normal_matrix(mu, sigma, Z)
                scalar_mul(Z, sigma)
                scalar_add(Z, mu)
                scalar_exp(Z)
                update_stock(X, s, S, Z, n_restock)

        X.print()

fn compute_freq(x_init: Float64, 
                firm: Firm, 
                num_firms: Int, 
                sim_length: Int) -> Float64:
        
        var firm_counter: Int = 0

        @parameter
        fn calc_firm(num_firms: Int):
                var x = x_init
                var restock_counter = 0 
                
                for t in range(sim_length):
                        let Z: Float64 = randn_float64()
                        let D: Float64 = exp(firm.mu + firm.sigma*Z)
                        if x <= firm.s:
                                x = max(firm.S - D, 0)
                                restock_counter += 1
                        else:
                                x = max(x - D, 0)

                if restock_counter > 1:
                        firm_counter += 1
        parallelize[calc_firm](num_firms)
        return firm_counter / num_firms


fn compute_freq_loop(x_init: Float64, 
                firm: Firm, 
                num_firms: Int, 
                sim_length: Int) -> Float64:
        
        var firm_counter: Int = 0

        for i in range(num_firms):
                var x = x_init
                var restock_counter = 0 
                
                for t in range(sim_length):
                        let Z: Float64 = randn_float64()
                        let D: Float64 = exp(firm.mu + firm.sigma*Z)
                        if x <= firm.s:
                                x = max(firm.S - D, 0)
                                restock_counter += 1
                        else:
                                x = max(x - D, 0)

                if restock_counter > 1:
                        firm_counter += 1
        return firm_counter / num_firms

fn main():
        let firm: Firm = Firm(s=10, S=100, mu=1.0, sigma=0.5)
        let x_init: Int = 70
        let num_firms: Int = 500_000
        let sim_length: Int = 50

        let eval_begin: Float64 = now()
        print(compute_freq_loop(x_init, firm, num_firms, sim_length))
        let eval_end: Float64 = now()

        let execution_time = Float64((eval_end - eval_begin)) / 1e6
        print("Completed inventory dynamics calculation in ", execution_time, "ms")
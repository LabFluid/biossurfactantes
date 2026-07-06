import numpy as np
import matplotlib.pyplot as plt
import scipy
import sympy as sp


def ODE_system_solver(f, y0, t0, final_t, step = 1e-4):

    ts = np.arange(t0, final_t, step)
    ys = np.empty((len(ts), len(y0)), dtype=np.float32)
    ys[0] = y0

    for i, t in enumerate(ts[:-1]):
        ys[i + 1] = ys[i] + f(t, ys[i]) * step

    return ts, ys



mu_w = 1e-3
mu_g = 1e-5
S_wc = 0.2
S_gr = 0.1
S_left = 0.9

def k_rw(S):
	return ((S - S_wc) / (1 - S_wc - S_gr)) ** 4

def k_rg(S):
	return ((1 - S - S_gr) / (1 - S_wc - S_gr)) ** 2

def f(S):
	lambda_w = k_rw(S) / mu_w
	lambda_g = k_rg(S) / mu_g
	return lambda_w / (lambda_w + lambda_g)

def rhs(t, y, dx = 0.01):
    F = np.empty(y.shape[0] + 1, dtype=np.float32)
    F[0] = f(S_left)
    F[1:] = f(y)

    return -(F[1:] - F[:-1]) / dx


S = sp.Symbol("S", real=True)
f_prime = sp.lambdify(S, sp.diff(f(S), S), "numpy")
f_second = sp.lambdify(S, sp.diff(f(S), S, 2), "numpy")


def S0(x):
	return 0.8 * x

def get_sol(x, t):

    def obj(xi):
        return xi + f_prime(S0(xi)) * t - x

    def obj_prime(xi):
    	return 1 + f_second(S0(xi)) * 0.8 * t

    res = scipy.optimize.root_scalar(obj, x0=x, fprime=obj_prime, method="newton")

    return S0(res.root)







Nx = 100
dx = 1.0 / Nx
dt = 0.00025
T_max = 0.75

xs = np.linspace(0.0, 1.0, Nx)
S_inicial = S0(xs)

ts, us = ODE_system_solver(lambda t, x: rhs(t, x, dx=dx), S_inicial, 0.0, T_max, dt)



idx = Nx - 1
x = xs[-1].item()
plt.plot(ts, us[:, idx], color="red")
plt.plot(ts, [get_sol(x, t) for t in ts], color="black", linestyle="-.")


plt.xlabel('t')
plt.ylabel('Saturação S')
plt.grid(True)



plt.show()
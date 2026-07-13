import numpy as np
import matplotlib.pyplot as plt
import scipy
import sympy as sp


def ODE_system_solver(f, y0, t0, final_t, step = 1e-4):

    ts = np.arange(t0, final_t, step)
    ys = np.empty((len(ts), len(y0)), dtype=np.float64)
    ys[0] = y0

    for i, t in enumerate(ts[:-1]):
        ys[i + 1] = ys[i] + f(t, ys[i]) * step

    return ts, ys



S_left = 1.0
S_i = 0.0
alpha = 0.0

'''
mu_w = 1e-3
mu_g = 1e-5
S_wc = 0.2
S_gr = 0.1

def k_rw(S):
    return ((S - S_wc) / (1 - S_wc - S_gr)) ** 4

def k_rg(S):
    return ((1 - S - S_gr) / (1 - S_wc - S_gr)) ** 2

def f(S):
    lambda_w = k_rw(S) / mu_w
    lambda_g = k_rg(S) / mu_g
    return lambda_w / (lambda_w + lambda_g)
'''

M = 2
def f(S):
    return S ** 2 / (S ** 2 + M * (1 - S) ** 2)


def rhs(t, y, dx = 0.01):
    F = np.empty(y.shape[0] + 1, dtype=np.float64)
    F[0] = f(S_left)
    F[1:] = f(y)

    return -(F[1:] - F[:-1]) / dx


S = sp.Symbol("S", real=True)
f_prime = sp.lambdify(S, sp.diff(f(S), S), "numpy")
f_second = sp.lambdify(S, sp.diff(f(S), S, 2), "numpy")
S_inflection = scipy.optimize.bisect(f_second, 0.0, 0.9)

print(S_inflection)
print(f_prime(S_left))

def f_prime_inv(f):

    def obj(S):
        return f - f_prime(S)

    return scipy.optimize.brentq(obj, S_inflection, S_left)


def get_S_R(S_L):
    if S_L >= S_inflection:
        return S_L

    return scipy.optimize.brentq(lambda S: f_prime(S) - (f(S) - f(S_L)) / (S - S_L), S_inflection, S_left)

def get_shock_vel(S_L, S_R=None):
    if S_R is None:
        S_R = get_S_R(S_L)
    return (f(S_L) - f(S_R)) / (S_L - S_R)


def S0(x):
	return x * alpha + S_i

def get_sol_after_shock(x, t):

    def obj(xi):
        return xi + f_prime(S0(xi)) * t - x

    def obj_prime(xi):
    	return 1 + f_second(S0(xi)) * alpha * t

    res = scipy.optimize.root_scalar(obj, x0=x, fprime=obj_prime, method="newton")

    return S0(res.root)




def get_sol(x, t, x_shock):
    xi = x / t

    if xi < f_prime(S_left): # antes da rarefação
        return S_left
    elif x < x_shock: # na rarefação
        return f_prime_inv(xi)
    else: # depois do choque
        return get_sol_after_shock(x, t)




Nx = 200
dx = 1.0 / Nx
dt = 0.00001
T_max = 0.75

xs = np.linspace(0.0, 1.0, Nx)

ts, us = ODE_system_solver(lambda t, x: rhs(t, x, dx=dx), S0(xs), 0.0, T_max, dt)





shock_pos = scipy.integrate.solve_ivp(lambda t, x: get_shock_vel(get_sol_after_shock(x[0], t)) if x <= 1 else 0, (0, T_max), [0], t_eval=ts).y[0]




idx_step = len(ts) // 6
for idx, t in enumerate(ts[idx_step::idx_step], start=1):
    plt.plot(xs, us[idx * idx_step, :], color="red")

for idx, (t, x_shock) in enumerate(zip(ts[idx_step::idx_step], shock_pos[idx_step::idx_step])):
    S_ahead = get_sol_after_shock(x_shock, t)
    S_behind = get_S_R(S_ahead)

    plt.scatter([x_shock], [S_ahead], color="red")
    plt.scatter([x_shock], [S_behind], color="blue")
    plt.plot(xs, [get_sol(x, t, x_shock) for x in xs], color="black", linestyle="-.")
    # plt.plot(xs, [get_S_R(get_sol(x, t, x_shock)) for x in xs], color="black", linestyle="--")



plt.xlabel('x')
plt.ylabel('Saturação S')
plt.grid(True)



plt.show()
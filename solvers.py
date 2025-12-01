import numpy as np

import numpy as np


def step_ssprk33(rhs_func, u, t, dt):

    # Estágio 1
    k1 = rhs_func(t, u)
    u1 = u + dt * k1
    
    # Estágio 2
    k2 = rhs_func(t + dt, u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
    
    # Estágio 3
    k3 = rhs_func(t + 0.5 * dt, u2)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * k3)
    
    return u_new

def step_rk4(L_func, u_n, t_n, dt):

    # k1
    k1 = L_func(t_n, u_n)
    
    # k2
    k2 = L_func(t_n + 0.5 * dt, u_n + 0.5 * dt * k1)
    
    # k3
    k3 = L_func(t_n + 0.5 * dt, u_n + 0.5 * dt * k2)
    
    # k4
    k4 = L_func(t_n + dt, u_n + dt * k3)
    
    # Resultado final
    u_n_plus_1 = u_n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return u_n_plus_1

def step_euler(L_func, u_n, t_n, dt):
    return u_n + dt * L_func(t_n, u_n)
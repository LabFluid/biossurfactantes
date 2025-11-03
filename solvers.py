import numpy as np

def step_ssprk33(L_func, u_n, t_n, dt):
    
    # Estágio 1
    u_1 = u_n + dt * L_func(t_n, u_n)
    
    # Estágio 2
    u_2 = (0.75 * u_n) + (0.25 * u_1) + (0.25 * dt * L_func(t_n + dt, u_1))

    # Estágio 3 
    u_n_plus_1 = (1.0/3.0 * u_n) + (2.0/3.0 * u_2) + (2.0/3.0 * dt * L_func(t_n + 0.5 * dt, u_2))
    
    return u_n_plus_1

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
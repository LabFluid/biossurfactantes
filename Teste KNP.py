import numpy as np
import matplotlib.pyplot as plt

nx = 400
L = 1.0
dx = L / nx
gamma = 1.4
t_final = 0.01
cfl = 0.5
x = np.linspace(dx/2, L - dx/2, nx)

def funcao_fluxo(rho, v, p):
    E = p / (gamma - 1) + 0.5 * rho * v**2
    return np.array([rho * v, rho * v**2 + p, v * (E + p)])

def vel_onda(rho, v, p):
    c = np.zeros_like(rho)
    c = np.sqrt(gamma * p/rho)
    return v, c

def minmod(a, b, c):
    sinal = (np.sign(a) == np.sign(b)) & (np.sign(b) == np.sign(c))
    return np.where(sinal, np.sign(a) * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0)

P = np.zeros((3, nx))
cond_esq = x < 0.1
cond_meio = (x >= 0.1) & (x < 0.9)
cond_dir = x >= 0.9

P[0, cond_esq] = 1.0;     P[1, cond_esq] = 0.0;     P[2, cond_esq] = 1000.0
P[0, cond_meio] = 1.0;    P[1, cond_meio] = 0.0;    P[2, cond_meio] = 0.01
P[0, cond_dir] = 1.0;     P[1, cond_dir] = 0.0;     P[2, cond_dir] = 100.0

t = 0.0

while t < t_final:
    
    v_all, c_all = vel_onda(P[0, :], P[1, :], P[2, :])
    max_speed = np.max(np.abs(v_all) + c_all)
    dt = cfl * dx / max_speed
    if t + dt > t_final: dt = t_final - t
    
    P_ext = np.pad(P, ((0, 0), (2, 2)), mode='constant')
    
    P_ext[0, 1] = P_ext[0, 2];  P_ext[0, 0] = P_ext[0, 3];  P_ext[0, -2] = P_ext[0, -3];  P_ext[0, -1] = P_ext[0, -4]
    P_ext[1, 1] = -P_ext[1, 2]; P_ext[1, 0] = -P_ext[1, 3]; P_ext[1, -2] = -P_ext[1, -3]; P_ext[1, -1] = -P_ext[1, -4]
    P_ext[2, 1] = P_ext[2, 2];  P_ext[2, 0] = P_ext[2, 3];  P_ext[2, -2] = P_ext[2, -3];  P_ext[2, -1] = P_ext[2, -4]
    
    H = np.zeros((3, nx + 1))

    for j in range(nx + 1):
        P_jm1, P_j0, P_j1, P_j2 = P_ext[:, j], P_ext[:, j+1], P_ext[:, j+2], P_ext[:, j+3]

        theta = 1.0
        slope_left  = minmod(theta*(P_j0 - P_jm1)/dx, (P_j1 - P_jm1)/(2*dx), theta*(P_j1 - P_j0)/dx)
        slope_right = minmod(theta*(P_j1 - P_j0)/dx, (P_j2 - P_j0)/(2*dx), theta*(P_j2 - P_j1)/dx)

        P_neg = P_j0 + slope_left * dx / 2.0
        P_pos  = P_j1 - slope_right * dx / 2.0
        
        v_neg, c_neg = vel_onda(P_neg[0], P_neg[1], P_neg[2])
        v_pos, c_pos = vel_onda(P_pos[0], P_pos[1], P_pos[2])
        
        a_pos  = max(v_neg + c_neg, v_pos + c_pos, 0)
        a_neg = min(v_neg - c_neg, v_pos - c_pos, 0)  
      
        flux_neg = funcao_fluxo(P_neg[0], P_neg[1], P_neg[2])
        flux_pos  = funcao_fluxo(P_pos[0], P_pos[1], P_pos[2])
        
        E_neg = P_neg[2]/(gamma-1) + 0.5*P_neg[0]*P_neg[1]**2
        E_pos = P_pos[2]/(gamma-1) + 0.5*P_pos[0]*P_pos[1]**2
        U_neg = np.array([P_neg[0], P_neg[0]*P_neg[1], E_neg])
        U_pos  = np.array([P_pos[0],  P_pos[0]*P_pos[1],  E_pos])
        
        H[:, j] = (a_pos*flux_neg - a_neg*flux_pos + a_pos*a_neg*(U_pos - U_neg)) / (a_pos - a_neg)

    rho, v, p = P[0,:], P[1,:], P[2,:]
    E = p / (gamma - 1) + 0.5 * rho * v**2
    U = np.array([rho, rho * v, E])
    
    U = U - (dt / dx) * (H[:, 1:] - H[:, :-1])
    
    rho_novo = U[0,:]
    v_novo = U[1,:] / rho_novo
    p_novo = (gamma - 1) * (U[2,:] - 0.5 * rho_novo * v_novo**2)
    P = np.array([rho_novo, v_novo, p_novo])
    
    t += dt

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
titles = ['Densidade ($\\rho$)', 'Velocidade ($v$)', 'Pressão ($p$)']
for i, ax in enumerate(axes):
    ax.plot(x, P[i, :])
    ax.set_title(titles[i])
    ax.set_xlim(0, L)
    ax.grid(True)

fig.suptitle(f'Solução das equações de Euler em t = {t:.2f}s (KNP)')
plt.show()
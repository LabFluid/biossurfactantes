import numpy as np
import matplotlib.pyplot as plt

# Parametros
GAMMA = 1.4
CFL = 0.475
T_FINAL = 0.01

NX, NY = 400, 100
LX, LY = 1.0, 0.25
DX, DY = LX/NX, LY/NY

x = np.linspace(DX/2, LX-DX/2, NX)
y = np.linspace(DY/2, LY-DY/2, NY)
X, Y = np.meshgrid(x, y, indexing='ij')


#Funções
def cons_to_prim(U):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = np.maximum((GAMMA-1) * (U[3] - 0.5*rho*(u**2 + v**2)), 1e-10)
    return rho, u, v, p

def prim_to_cons(rho, u, v, p):
    E = p/(GAMMA-1) + 0.5*rho*(u**2 + v**2)
    return np.array([rho, rho*u, rho*v, E])

def flux_x(rho, u, v, p):
    E = p/(GAMMA-1) + 0.5*rho*(u**2 + v**2)
    return np.array([rho*u, rho*u**2 + p, rho*u*v, u*(E+p)])

def flux_y(rho, u, v, p):
    E = p/(GAMMA-1) + 0.5*rho*(u**2 + v**2)
    return np.array([rho*v, rho*u*v, rho*v**2 + p, v*(E+p)])

def sound_speed(rho, p):
    return np.sqrt(GAMMA * p / rho)

def minmod(a, b, c):
    result = np.zeros_like(a)
    pos = (a > 0) & (b > 0) & (c > 0)
    neg = (a < 0) & (b < 0) & (c < 0)
    result[pos] = np.minimum(np.minimum(a[pos], b[pos]), c[pos])
    result[neg] = np.maximum(np.maximum(a[neg], b[neg]), c[neg])
    return result

def reconstruct(U, dx, axis):
    theta = 2

    if axis == 0:  # direção x
        dU_L = (U[:, 1:-1, :] - U[:, :-2, :]) / dx
        dU_C = (U[:, 2:, :] - U[:, :-2, :]) / (2*dx)
        dU_R = (U[:, 2:, :] - U[:, 1:-1, :]) / dx
        slope = minmod(theta*dU_L, dU_C, theta*dU_R)
        U_L = U[:, 1:-2, :] + slope[:, :-1, :] * dx/2
        U_R = U[:, 2:-1, :] - slope[:, 1:, :] * dx/2
    else:  # direção y
        dU_L = (U[:, :, 1:-1] - U[:, :, :-2]) / dx
        dU_C = (U[:, :, 2:] - U[:, :, :-2]) / (2*dx)
        dU_R = (U[:, :, 2:] - U[:, :, 1:-1]) / dx
        slope = minmod(theta*dU_L, dU_C, theta*dU_R)
        U_L = U[:, :, 1:-2] + slope[:, :, :-1] * dx/2
        U_R = U[:, :, 2:-1] - slope[:, :, 1:] * dx/2
    
    return U_L, U_R

def knp_flux_x(U):
    U_L, U_R = reconstruct(U, DX, axis=0)
    
    rho_L, u_L, v_L, p_L = cons_to_prim(U_L)
    rho_R, u_R, v_R, p_R = cons_to_prim(U_R)
    
    c_L, c_R = sound_speed(rho_L, p_L), sound_speed(rho_R, p_R)
    a_plus = np.maximum(np.maximum(u_L + c_L, u_R + c_R), 0)
    a_minus = np.minimum(np.minimum(u_L - c_L, u_R - c_R), 0)
    
    F_L = flux_x(rho_L, u_L, v_L, p_L)
    F_R = flux_x(rho_R, u_R, v_R, p_R)
    
    H = (a_plus * F_L - a_minus * F_R + a_plus * a_minus * (U_R - U_L)) / (a_plus - a_minus)
    
    return H


def knp_flux_y(U):
    U_L, U_R = reconstruct(U, DY, axis=1)
    
    rho_L, u_L, v_L, p_L = cons_to_prim(U_L)
    rho_R, u_R, v_R, p_R = cons_to_prim(U_R)
    
    c_L, c_R = sound_speed(rho_L, p_L), sound_speed(rho_R, p_R)
    a_plus = np.maximum(np.maximum(v_L + c_L, v_R + c_R), 0)
    a_minus = np.minimum(np.minimum(v_L - c_L, v_R - c_R), 0)
    
    G_L = flux_y(rho_L, u_L, v_L, p_L)
    G_R = flux_y(rho_R, u_R, v_R, p_R)
    
    H = (a_plus * G_L - a_minus * G_R + a_plus * a_minus * (U_R - U_L)) / (a_plus - a_minus)
    
    return H

def compute_rhs(U):
    Hx = knp_flux_x(U)
    Hy = knp_flux_y(U)
    
    L = np.zeros_like(U)
    L[:, 2:-2, :] -= (Hx[:, 1:, :] - Hx[:, :-1, :]) / DX
    L[:, :, 2:-2] -= (Hy[:, :, 1:] - Hy[:, :, :-1]) / DY
    
    return L

def apply_bc(U):
#Condições de contorno para x
    U[:, 0, :] = U[:, 2, :]
    U[:, 1, :] = U[:, 2, :]
    U[1, 0, :] = -U[1, 2, :]  
    U[1, 1, :] = -U[1, 2, :]
    
    U[:, -1, :] = U[:, -3, :]
    U[:, -2, :] = U[:, -3, :]
    U[1, -1, :] = -U[1, -3, :] 
    U[1, -2, :] = -U[1, -3, :]

#Condições de contorno para y
    U[:, :, 0] = U[:, :, 2]
    U[:, :, 1] = U[:, :, 2]
    U[2, :, 0] = -U[2, :, 2]
    U[2, :, 1] = -U[2, :, 2]
    
    U[:, :, -1] = U[:, :, -3]
    U[:, :, -2] = U[:, :, -3]
    U[2, :, -1] = -U[2, :, -3]
    U[2, :, -2] = -U[2, :, -3]
    return U

def ssprk3(U, dt):
    U1 = apply_bc(U + dt * compute_rhs(U))
    U2 = apply_bc(0.75*U + 0.25*(U1 + dt * compute_rhs(U1)))
    return apply_bc(U/3 + 2/3*(U2 + dt * compute_rhs(U2)))


def dt_cfl(U):
    rho, u, v, p = cons_to_prim(U)
    c = sound_speed(rho, p)
    max_speed = np.max(np.abs(u) + np.abs(v) + c)
    return CFL * min(DX, DY) / (max_speed)

#Condição inicial
def initial_condition():
    rho = np.ones((NX, NY))
    u = np.zeros((NX, NY))
    v = np.zeros((NX, NY))
    p = np.ones((NX, NY)) * 0.01
    
    p[X<0.1] = 1000.0
    p[X >= 0.9]= 100.0
    
    return prim_to_cons(rho, u, v, p)

U = apply_bc(initial_condition())
t, step = 0.0, 0
    
while t < T_FINAL:
    dt = min(dt_cfl(U), T_FINAL - t)
    U = ssprk3(U, dt)
    t += dt
    step += 1
    if step % 200 == 0:
        print(f"passo = {step}  t = {t:.5f}, dt = {dt:.2e}")

def plot_results(U, T_FINAL):
    rho, u, v, p = cons_to_prim(U)
    
    rho_1d = rho[:, 1]
    u_1d = u[:, 1]
    p_1d = p[:, 1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(x, rho_1d, 'x','b-', lw=1.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('ρ')
    axes[0].set_title('Densidade')
    axes[0].grid(True)
    axes[0].set_xlim(0, LX)
    
    axes[1].plot(x, u_1d, 'r-', lw=1.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].set_title('Velocidade')
    axes[1].grid(True)
    axes[1].set_xlim(0, LX)
    
    axes[2].plot(x, p_1d, 'g-')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('p')
    axes[2].set_title('Pressão')
    axes[2].grid(True)
    axes[2].set_xlim(0, LX)
    
    fig.suptitle(f'Euler 2D→1D, t = {T_FINAL}', fontsize=12)
    plt.savefig('euler_2d-1d.png')
    
    return rho_1d, u_1d, p_1d

rho_1d, u_1d, p_1d = plot_results(U, T_FINAL)

# Teste de plot 2D
plt.figure(figsize=(8,6))
rho = cons_to_prim(U)[0]
plt.pcolormesh(X, Y, rho)
plt.show()
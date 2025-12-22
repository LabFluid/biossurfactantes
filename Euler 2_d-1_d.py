import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
GAMMA = 1.4
CFL = 0.475
T_FINAL = 0.01

NX, NY = 400, 100  
LX, LY = 1.0, 1.0
DX, DY = LX/NX, LY/NY

x = np.linspace(DX/2, LX-DX/2, NX)
y = np.linspace(DY/2, LY-DY/2, NY)
X, Y = np.meshgrid(x, y, indexing='ij')

def cons_to_prim(U_vec):
    rho = U_vec[0]
    u = U_vec[1] / rho
    v = U_vec[2] / rho
    p = (GAMMA-1) * (U_vec[3] - 0.5*rho*(u**2 + v**2))
    return rho, u, v, p

def prim_to_cons(rho, u, v, p):
    E = p/(GAMMA-1) + 0.5*rho*(u**2 + v**2)
    return np.array([rho, rho*u, rho*v, E])

def flux_x_scalar(rho, u, v, p):
    E = p/(GAMMA-1) + 0.5*rho*(u**2 + v**2)
    return np.array([rho*u, rho*u**2 + p, rho*u*v, u*(E+p)])

def flux_y_scalar(rho, u, v, p):
    E = p/(GAMMA-1) + 0.5*rho*(u**2 + v**2)
    return np.array([rho*v, rho*u*v, rho*v**2 + p, v*(E+p)])

def sound_speed(rho, p):
    return np.sqrt(GAMMA * np.maximum(p, 1e-20) / np.maximum(rho, 1e-20))

def minmod(a, b, c):
    result = np.zeros_like(a)
    pos = (a > 0) & (b > 0) & (c > 0)
    neg = (a < 0) & (b < 0) & (c < 0)
    result[pos] = np.minimum(np.minimum(a[pos], b[pos]), c[pos])
    result[neg] = np.maximum(np.maximum(a[neg], b[neg]), c[neg])
    return result

def reconstruct(U, dx, axis):
    theta = 2 
    
    if axis == 0:
        du_L = (U[:, 1:-1, :] - U[:, :-2, :]) / dx
        du_R = (U[:, 2:, :] - U[:, 1:-1, :]) / dx
        du_C = (U[:, 2:, :] - U[:, :-2, :]) / (2*dx)
        
        slope = minmod(theta*du_L, du_C, theta*du_R)
        
        U_face_L = U[:, 1:-1, :] - (slope * dx/2)
        U_face_R = U[:, 1:-1, :] + (slope * dx/2)
        return U_face_R[:, :-1, :], U_face_L[:, 1:, :]
    else:
        du_L = (U[:, :, 1:-1] - U[:, :, :-2]) / dx
        du_R = (U[:, :, 2:] - U[:, :, 1:-1]) / dx
        du_C = (U[:, :, 2:] - U[:, :, :-2]) / (2*dx)
        
        slope = minmod(theta*du_L, du_C, theta*du_R)

        U_face_L = U[:, :, 1:-1] - (slope * dx/2)
        U_face_R = U[:, :, 1:-1] + (slope * dx/2)

        return U_face_R[:, :, :-1], U_face_L[:, :, 1:]

def knp_flux_x(U):
    U_L_faces, U_R_faces = reconstruct(U, DX, axis=0)
    _, nx_inter, ny = U_L_faces.shape
    H = np.zeros((4, nx_inter, ny))

    for i in range(nx_inter):
        for j in range(ny):
            uL_vec = U_L_faces[:, i, j]
            uR_vec = U_R_faces[:, i, j]

            rL, uL, vL, pL = cons_to_prim(uL_vec)
            rR, uR, vR, pR = cons_to_prim(uR_vec)

            cL = np.sqrt(GAMMA * np.maximum(pL, 1e-20) / np.maximum(rL, 1e-20))
            cR = np.sqrt(GAMMA * np.maximum(pR, 1e-20) / np.maximum(rR, 1e-20))

            ap = max(uL + cL, uR + cR, 0.0)
            am = min(uL - cL, uR - cR, 0.0)

            fL = flux_x_scalar(rL, uL, vL, pL)
            fR = flux_x_scalar(rR, uR, vR, pR)

            H[:, i, j] = (ap * fL - am * fR + ap * am * (uR_vec - uL_vec)) / (ap-am)

    return H

def knp_flux_y(U):
    U_L_faces, U_R_faces = reconstruct(U, DY, axis=1)
    _, nx, ny_inter = U_L_faces.shape
    G = np.zeros((4, nx, ny_inter))

    for i in range(nx):
        for j in range(ny_inter):
            uL_vec = U_L_faces[:, i, j]
            uR_vec = U_R_faces[:, i, j]

            rL, uL, vL, pL = cons_to_prim(uL_vec)
            rR, uR, vR, pR = cons_to_prim(uR_vec)

            cL = np.sqrt(GAMMA * np.maximum(pL, 1e-20) / np.maximum(rL, 1e-20))
            cR = np.sqrt(GAMMA * np.maximum(pR, 1e-20) / np.maximum(rR, 1e-20))

            ap = max(vL + cL, vR + cR, 0.0)
            am = min(vL - cL, vR - cR, 0.0)

            gL = flux_y_scalar(rL, uL, vL, pL)
            gR = flux_y_scalar(rR, uR, vR, pR)

            G[:, i, j] = (ap * gL - am * gR + ap * am * (uR_vec - uL_vec)) / (ap-am)

    return G

def compute_rhs(U):
    Hx = knp_flux_x(U)
    Hy = knp_flux_y(U)
    
    L = np.zeros_like(U)
    
    L[:, 2:-2, :] -= (Hx[:, 1:, :] - Hx[:, :-1, :]) / DX
    L[:, :, 2:-2] -= (Hy[:, :, 1:] - Hy[:, :, :-1]) / DY

    return L

def apply_bc(U):
    # Condições de contorno para x
    U[:, 0, :] = U[:, 2, :]
    U[:, 1, :] = U[:, 2, :]
    U[1, 0, :] = -U[1, 2, :]
    U[1, 1, :] = -U[1, 2, :]
    
    U[:, -1, :] = U[:, -3, :]
    U[:, -2, :] = U[:, -3, :]
    U[1, -1, :] = -U[1, -3, :]
    U[1, -2, :] = -U[1, -3, :]

    # Condições de contorno para y
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
    return CFL * min(DX, DY) / max_speed

def initial_condition():
    rho = np.ones((NX, NY))
    u = np.zeros((NX, NY))
    v = np.zeros((NX, NY))
    p = np.ones((NX, NY)) * 0.01  
    
    p[X < 0.1] = 1000.0
    p[X >= 0.9] = 100.0
    
    return prim_to_cons(rho, u, v, p)

U = apply_bc(initial_condition())
t, step = 0.0, 0

while t < T_FINAL:
    dt = min(dt_cfl(U), T_FINAL - t)
    U = ssprk3(U, dt)
    t += dt
    step += 1
    if step % 5 == 0:
        print(f"Passo: {step} | Tempo: {t:.5f} | dt: {dt:.2e}")

def plot_results(U, T_FINAL):
    rho, u, v, p = cons_to_prim(U)
    
    j_mid = NY // 2
    rho_1d = rho[:, j_mid]
    u_1d = u[:, j_mid]
    p_1d = p[:, j_mid]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(x, rho_1d, 'b-', lw=1.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('ρ')
    axes[0].set_title('Densidade')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, LX)
    
    axes[1].plot(x, u_1d, 'r-', lw=1.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].set_title('Velocidade')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, LX)
    
    axes[2].plot(x, p_1d, 'g-', lw=1.5)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('p')
    axes[2].set_title('Pressão')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, LX)
    
    fig.suptitle(f'Euler 2D - Corte em y - t = {T_FINAL}', fontsize=12)
    plt.savefig('euler_2d_corte1d.png', dpi=150)
    return rho_1d, u_1d, p_1d

def plot_2d(U, T_FINAL):
    rho, u, v, p = cons_to_prim(U)
    
    x_edges = np.linspace(0, LX, NX+1)
    y_edges = np.linspace(0, LY, NY+1)
    X_edges, Y_edges = np.meshgrid(x_edges, y_edges, indexing='ij')
    fig = plt.figure(figsize=(8, 6))
    
    im1 = plt.pcolormesh(X_edges, Y_edges, rho, shading='flat', cmap='viridis')
    plt.title('Densidade (ρ)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(im1)
    fig.suptitle(f'Euler 2D (KNP) - t = {T_FINAL}', fontsize=14)
    plt.savefig('euler_2d.png', dpi=150)

rho_1d, u_1d, p_1d = plot_results(U, T_FINAL)
plot_2d(U, T_FINAL)

plt.show()
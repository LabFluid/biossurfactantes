import matplotlib.pyplot as plt
import numpy as np
from reservatório import carregar_dados_reservatorio

# Parâmetros
muw = 1.0e-3   #viscosidade da água
mug = 2.0e-5   #viscosidade do gás 
Swc = 0.2  # saturação residual de água
Sgr = 0.0  # saturação residual de gás
Sstar = 0.37     #saturação crítica de água

nmax = 8.0e13  # densidade máxima de espuma
u_ = 3.0e-5    # velocidade de injeção
Sw_0 = 1.0     # saturação inicial de água
Sw_ = 0.372   # saturação de água injetada
nd_0 = 0.0     # densidade inicial da espuma
nd_ = 0.0     # densidade da espuma injetada
phi = 0.2      # porosidade

A_foam = 400.0  # coeficiente de formação de espuma
Kc = 1.0e-6    # constante temporal

Lx = 3.67   # comprimento em x
Ly = 1.0    # comprimento em y
nx = 220    # número de células em x
ny = 60     # número de células em y

t_final = 1000.0 # tempo final
ux = 3.0e-5    # velocidade de propagação em x 
uy = 0       # velocidade de propagação em y
theta = 1.0   # parâmetro do minmod
cfl = 0.5      # parâmetro cfl

tol = 1.0e-6  # tolerância

# malha
dx = Lx/nx
dy = Ly/ny
x = np.linspace(dx/2, Lx-dx/2, nx) # centros das células em x
y = np.linspace(dy/2, Ly-dy/2, ny) # centros das células em y
X, Y = np.meshgrid(x, y, indexing='ij')

# Funções
def Sg(Sw):
    return 1 - Sw

def nd_le(Sw):
    return np.where(Sw < Sstar, 0.0, np.tanh(A_foam * (Sw - Sstar)))

def krw(Sw):
    return np.maximum(0, ((Sw - Swc)/(1 - Swc - Sgr + 1e-15))**4)

def krg(Sw, nD):
    krg0 = np.maximum(0, ((1 - Sw - Sgr)/(1 - Swc - Sgr + 1e-15))**2)
    return krg0 / (18500*nD + 1)

def lambda_w(Sw): 
    return krw(Sw)/muw

def lambda_g(Sw, nD): 
    return krg(Sw, nD)/mug

def fw(Sw, nD):
    lw = lambda_w(Sw)
    lg = lambda_g(Sw, nD)
    return lw/(lw + lg + 1e-20)

def fg(Sw, nD):
    return 1 - fw(Sw, nD)

def minmod(a, b, c):
    result = np.zeros_like(a)
    pos = (a > 0) & (b > 0) & (c > 0)
    neg = (a < 0) & (b < 0) & (c < 0)
    result[pos] = np.minimum(np.minimum(a[pos], b[pos]), c[pos])
    result[neg] = np.maximum(np.maximum(a[neg], b[neg]), c[neg])
    return result

def reconstruct(S, axis): 
    if axis == 0:  # direção x
        Sw_pad = np.pad(S, ((0,0), (1,1), (0,0)), mode='edge')
        dSw_L = (Sw_pad[:, 1:-1, :] - Sw_pad[:, :-2, :]) / dx
        dSw_R = (Sw_pad[:, 2:, :] - Sw_pad[:, 1:-1, :]) / dx
        dSw_C = (Sw_pad[:, 2:, :] - Sw_pad[:, :-2, :]) / (2*dx)
        
        slope = minmod(theta*dSw_L, dSw_C, theta*dSw_R)

        Sw_L = S[:, :-1, :] + slope[:, :-1, :] * dx/2
        Sw_R = S[:, 1:, :] - slope[:, 1:, :] * dx/2
        return Sw_L, Sw_R
    else:  # direção y
        Sw_pad = np.pad(S, ((0,0), (0,0), (1,1)), mode='edge')
        dSw_L = (Sw_pad[:, :, 1:-1] - Sw_pad[:, :, :-2]) / dy
        dSw_R = (Sw_pad[:, :, 2:] - Sw_pad[:, :, 1:-1]) / dy
        dSw_C = (Sw_pad[:, :, 2:] - Sw_pad[:, :, :-2]) / (2*dy)
        
        slope = minmod(theta*dSw_L, dSw_C, theta*dSw_R)

        Sw_L = S[:, :, :-1] + slope[:, :, :-1] * dy/2
        Sw_R = S[:, :, 1:] - slope[:, :, 1:] * dy/2
        return Sw_L, Sw_R
    
def flux_x(Sw, nD):
    return np.array([ux * fw(Sw, nD),
                     nD * fg(Sw, nD) * ux])

def flux_y(Sw, nD):
    return np.array([uy * fw(Sw, nD),
                     nD * fg(Sw, nD) * uy])

def initial_condition():
    S = np.zeros((2, nx, ny))
    S[0, :, :] = Sw_0
    S[1, :, :] = Sg(Sw_0) * nd_0
    return S

def apply_bc(S):
    # x = 0
    S[0, 0, :] = Sw_
    S[1, 0, :] = Sg(Sw_) * nd_
    
    return S

def eigvals(Sw, nD, u_face):
    eps = 1e-8
    dfw_dSw = (fw(Sw + eps, nD) - fw(Sw - eps, nD)) / eps
    l1 = u_face * dfw_dSw / phi
    Sg_val = Sg(Sw)
    l2 = np.where(Sg_val > 1e-10, (u_face * fg(Sw, nD)) / (phi * Sg_val), 0.0)
    return l1, l2

def KNP_flux_x(S):
    S_L, S_R = reconstruct(S, axis=0)
    n_faces = S_L.shape[1]
    H = np.zeros((2, n_faces, ny))
    
    for i in range(n_faces):
        for j in range(ny):
            Sw_L, Sw_R = S_L[0, i, j], S_R[0, i, j]
            Sg_L, Sg_R = Sg(Sw_L), Sg(Sw_R)

            nD_L = S_L[1, i, j] / Sg_L if Sg_L > 1e-15 else 0.0
            nD_R = S_R[1, i, j] / Sg_R if Sg_R > 1e-15 else 0.0
            nD_L = np.clip(nD_L, 0, 1)
            nD_R = np.clip(nD_R, 0, 1)

            u_face = ux

            l1_L, l2_L = eigvals(Sw_L, nD_L, u_face)
            l1_R, l2_R = eigvals(Sw_R, nD_R, u_face)
            
            a_plus = max(0, l1_L, l2_L, l1_R, l2_R)
            a_minus = min(0, l1_L, l2_L, l1_R, l2_R)

            hl = flux_x(Sw_L, nD_L)
            hr = flux_x(Sw_R, nD_R)
            
            s_l = S_L[:, i, j]
            s_r = S_R[:, i, j]

            denom = (a_plus - a_minus)
            if abs(denom) < 1e-12:
                H[:, i, j] = 0.5 * (hl + hr)
            else:
                H[:, i, j] = (a_plus * hl - a_minus * hr + (a_plus * a_minus) * (s_r - s_l)) / denom
    
    return H

def KNP_flux_y(S):
    if uy == 0: 
        return np.zeros((2, nx, ny-1))
    
    S_L, S_R = reconstruct(S, axis=1)
    n_faces = S_L.shape[2]
    H = np.zeros((2, nx, n_faces))
    
    for i in range(nx):
        for j in range(n_faces):
            Sw_L, Sw_R = S_L[0, i, j], S_R[0, i, j]
            Sg_L, Sg_R = Sg(Sw_L), Sg(Sw_R)
            
            nD_L = S_L[1, i, j] / Sg_L if Sg_L > 1e-15 else 0.0
            nD_R = S_R[1, i, j] / Sg_R if Sg_R > 1e-15 else 0.0
            nD_L = np.clip(nD_L, 0, 1)
            nD_R = np.clip(nD_R, 0, 1)

            u_face = uy

            l1_L, l2_L = eigvals(Sw_L, nD_L, u_face)
            l1_R, l2_R = eigvals(Sw_R, nD_R, u_face)
            
            a_plus = max(0, l1_L, l2_L, l1_R, l2_R)
            a_minus = min(0, l1_L, l2_L, l1_R, l2_R)

            hl = flux_y(Sw_L, nD_L)
            hr = flux_y(Sw_R, nD_R)
            
            s_l = S_L[:, i, j]
            s_r = S_R[:, i, j]

            denom = a_plus - a_minus
            if abs(denom) < 1e-12:
                H[:, i, j] = 0.5 * (hl + hr)
            else:
                H[:, i, j] = (a_plus * hl - a_minus * hr + (a_plus * a_minus) * (s_r - s_l)) / denom
    
    return H

def compute_rhs(S):
    Hx = KNP_flux_x(S) 
    Hy = KNP_flux_y(S)

    L = np.zeros_like(S)

    # Fluxo de entrada (x=0)
    Sw_in, nD_in = Sw_, nd_
    f_in = flux_x(Sw_in, nD_in)
    
    # Fluxo de saída (x=Lx)
    Sw_out = S[0, -1, :]
    Sg_out = Sg(Sw_out)
    nD_out = np.where(Sg_out > 1e-10, S[1, -1, :] / Sg_out, 0.0)
    f_out = np.zeros((2, ny))
    for j in range(ny):
        f_out[:, j] = flux_x(Sw_out[j], nD_out[j])
    
    # Primeira célula
    L[:, 0, :] = -(Hx[:, 0, :] - f_in[:, np.newaxis]) / dx
    # Células internas
    L[:, 1:-1, :] = -(Hx[:, 1:, :] - Hx[:, :-1, :]) / dx
    # Última célula
    L[:, -1, :] = -(f_out - Hx[:, -1, :]) / dx
    
    if uy != 0:
        L[:, :, 1:-1] -= (Hy[:, :, 1:] - Hy[:, :, :-1]) / dy
        # y = 0
        L[:, :, 0] -= Hy[:, :, 0]/ dy
        # y = Ly  
        L[:, :, -1] += Hy[:, :, -1] / dy
    
    # Espuma
    Sw_now = S[0]
    Sg_now = Sg(Sw_now)
    nD_now = np.where(Sg_now > 1e-10, S[1] / Sg_now, 0.0)
    fonte = np.zeros_like(S)
    tax_ger = Kc * nmax * (nd_le(Sw_now) - nD_now)
    fonte[0], fonte[1] = 0 , phi * Sg_now * (tax_ger)/nmax
    
    L = L / phi + fonte / phi
    
    return L

def ssprk3(U, dt):
    k1 = compute_rhs(U)
    U1 = apply_bc(U + dt * k1)
    
    k2 = compute_rhs(U1)
    U2 = apply_bc(0.75*U + 0.25*(U1 + dt * k2))
    
    k3 = compute_rhs(U2)
    return apply_bc(U/3 + 2/3*(U2 + dt * k3))

def dt_cfl(S):
    Sw = S[0]
    Sg_val = Sg(Sw)
    nD = np.where(Sg_val > 1e-10, S[1] / Sg_val, 0.0)
    l1, l2 = eigvals(Sw, nD, ux)
    max_speed_x = max(np.max(l1), np.max(l2))
    if uy != 0:
        l1y, l2y = eigvals(Sw, nD, uy)
        max_speed_y = max(np.max(l1y), np.max(l2y))
    else:
        max_speed_y = 0.0

    max_speed = max(max_speed_x, max_speed_y) / phi
    return cfl / (max_speed_x/dx + max_speed_y/dy)

# Loop temporal
S = apply_bc(initial_condition())
t, step = 0.0, 0

while t < t_final:
    dt = dt_cfl(S)
    if t + np.any(dt) > t_final:
        dt = t_final - t

    S = ssprk3(S, dt)

    t += dt
    step += 1

    if step % 5 == 0:
        print(f"Step: {step}, Time: {t:.2f}/{t_final}, dt: {dt:.4f}")

plt.figure(figsize=(15, 4))
plt.pcolormesh(X, Y, S[0], cmap='jet', vmin=Sw_, vmax=1.0)
plt.colorbar(label='Saturação de Água', )
plt.xlabel('x')
plt.ylabel('y')
plt.title('Saturação de Água no Meio Poroso')
plt.savefig("saturacao_final.png")
plt.show()
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from RT_FEM import Lx
from RT_FEM import Ly
from RT_FEM import create_darcy_data
from RT_FEM import nx
from RT_FEM import ny
from RT_FEM import solve_darcy
from RT_FEM import u_inj


muw = 1.0e-3
mug = 2.0e-5
Swc = 0.2
Sgr = 0.0
Sstar = 0.37
nmax = 8.0e13
fw_0 = 4.2867345113092153e-01
Sw_inj = 0.372
nd_0 = 0.0
nd_inj = 0.0
A_foam = 400.0
Kc = 1.0e-6
alpha_foam = 7.0e-16
foam_exponent = 2.0 / 3.0

dx = Lx / nx
dy = Ly / ny
phi_value = 0.2

t_final = 1000.0
dt_u = 10.0
theta = 1.0
cfl = 0.4
eps = 1.0e-12

output_dir = Path("output")


def create_coordinates():
    x = np.linspace(dx / 2, Lx - dx / 2, nx)
    y = np.linspace(dy / 2, Ly - dy / 2, ny)
    return np.meshgrid(x, y, indexing="ij")


def Sg(Sw):
    return 1.0 - Sw


def nd_le(Sw):
    return np.where(Sw < Sstar, 0.0, np.tanh(A_foam * (Sw - Sstar)))


def krw(Sw):
    Se = (Sw - Swc) / (1.0 - Swc - Sgr + eps)
    return np.maximum(0.0, Se) ** 4


def krg(Sw, nD):
    Sge = (1.0 - Sw - Sgr) / (1.0 - Swc - Sgr + eps)
    return np.maximum(0.0, Sge) ** 2


def lambda_w(Sw):
    return krw(Sw) / muw


def foam_viscosity(Sw, nD, u_mag=u_inj):
    if np.ndim(Sw) == 0 and np.ndim(nD) == 0 and np.ndim(u_mag) == 0:
        sw = float(Sw)
        nd = float(nD)
        u = abs(float(u_mag))
        sg = 1.0 - sw

        if nd <= 0.0 or u <= eps or sg <= eps:
            return mug

        denom = 1.0 - Swc - Sgr + eps
        se = (sw - Swc) / denom
        sge = (sg - Sgr) / denom
        if se <= 0.0 or sge <= 0.0:
            return mug

        krw_value = se**4
        krg_value = sge**2
        lw = krw_value / muw
        nf = nd * nmax
        beta = phi_value * sg / (krg_value * u + eps)
        A = alpha_foam * nf * lw * beta**foam_exponent
        a = A / 3.0
        b = (krg_value + lw * mug) / 2.0
        a3 = a * a * a
        sqrt_term = (2.0 * a3 * b + b * b) ** 0.5
        root = a + (a3 + b + sqrt_term) ** (1.0 / 3.0) + max(a3 + b - sqrt_term, 0.0) ** (1.0 / 3.0)

        return max(mug, mug + 3.0 * a * root * root / (lw + eps))

    Sw_arr, nD_arr, u_arr = np.broadcast_arrays(Sw, nD, u_mag)
    Sw_arr = Sw_arr.astype(np.float64, copy=False)
    nD_arr = nD_arr.astype(np.float64, copy=False)
    u_arr = np.abs(u_arr.astype(np.float64, copy=False))

    mu = np.full_like(Sw_arr, mug, dtype=np.float64)
    krg_value = krg(Sw_arr, nD_arr)
    lw = lambda_w(Sw_arr)
    sg = np.maximum(Sg(Sw_arr), 0.0)
    nf = np.maximum(nD_arr, 0.0) * nmax
    active = (nf > 0.0) & (u_arr > eps) & (krg_value > eps) & (lw > eps) & (sg > eps)

    if not np.any(active):
        return mu

    beta = phi_value * sg[active] / (krg_value[active] * u_arr[active] + eps)
    A = alpha_foam * nf[active] * lw[active] * beta**foam_exponent
    a = A / 3.0
    b = (krg_value[active] + lw[active] * mug) / 2.0
    a3 = a**3
    sqrt_term = np.sqrt(np.maximum(2.0 * a3 * b + b**2, 0.0))
    root = a + np.cbrt(a3 + b + sqrt_term) + np.cbrt(np.maximum(a3 + b - sqrt_term, 0.0))
    mu[active] = mug + 3.0 * a * root**2 / (lw[active] + eps)

    return np.maximum(mu, mug)


def lambda_g(Sw, nD, u_mag=u_inj):
    return krg(Sw, nD) / foam_viscosity(Sw, nD, u_mag)


def fw(Sw, nD, u_mag=u_inj):
    lw = lambda_w(Sw)
    lg = lambda_g(Sw, nD, u_mag)
    return lw / (lw + lg + eps)


def fg(Sw, nD, u_mag=u_inj):
    return 1.0 - fw(Sw, nD, u_mag)


def Sw_from_fw(fw_value, nD, u_mag=u_inj):
    lo = np.full_like(np.asarray(fw_value, dtype=np.float64), Swc, dtype=np.float64)
    hi = np.full_like(lo, 1.0 - Sgr - eps, dtype=np.float64)
    target = np.asarray(fw_value, dtype=np.float64)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = fw(mid, nD, u_mag) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)

    return 0.5 * (lo + hi)


def minmod(a, b, c):
    out = np.zeros_like(a)
    pos = (a > 0) & (b > 0) & (c > 0)
    neg = (a < 0) & (b < 0) & (c < 0)
    out[pos] = np.minimum(np.minimum(a[pos], b[pos]), c[pos])
    out[neg] = np.maximum(np.maximum(a[neg], b[neg]), c[neg])
    return out


def reconstruct_x(S):
    S_pad = np.pad(S, ((0, 0), (1, 1), (0, 0)), mode="edge")
    dL = (S_pad[:, 1:-1, :] - S_pad[:, :-2, :]) / dx
    dR = (S_pad[:, 2:, :] - S_pad[:, 1:-1, :]) / dx
    dC = (S_pad[:, 2:, :] - S_pad[:, :-2, :]) / (2.0 * dx)
    slope = minmod(theta * dL, dC, theta * dR)
    S_L = S[:, :-1, :] + 0.5 * dx * slope[:, :-1, :]
    S_R = S[:, 1:, :] - 0.5 * dx * slope[:, 1:, :]
    return S_L, S_R


def reconstruct_y(S):
    S_pad = np.pad(S, ((0, 0), (0, 0), (1, 1)), mode="edge")
    dL = (S_pad[:, :, 1:-1] - S_pad[:, :, :-2]) / dy
    dR = (S_pad[:, :, 2:] - S_pad[:, :, 1:-1]) / dy
    dC = (S_pad[:, :, 2:] - S_pad[:, :, :-2]) / (2.0 * dy)
    slope = minmod(theta * dL, dC, theta * dR)
    S_L = S[:, :, :-1] + 0.5 * dy * slope[:, :, :-1]
    S_R = S[:, :, 1:] - 0.5 * dy * slope[:, :, 1:]
    return S_L, S_R


def flux_x(Sw, nD, u):
    u_mag = np.abs(u)
    return np.array([u * fw(Sw, nD, u_mag), u * nD * fg(Sw, nD, u_mag)])


def flux_y(Sw, nD, v):
    v_mag = np.abs(v)
    return np.array([v * fw(Sw, nD, v_mag), v * nD * fg(Sw, nD, v_mag)])


def dfw_dSw(Sw, nD, u_mag=u_inj):
    h = 1.0e-7
    Swp = Sw + h
    Swm = Sw - h
    return (fw(Swp, nD, u_mag) - fw(Swm, nD, u_mag)) / (Swp - Swm + eps)


def eigvals(Sw, nD, vel, phi_cell):
    vel_mag = abs(vel)
    lambda_1 = vel * dfw_dSw(Sw, nD, vel_mag) / (phi_cell + eps)
    sg = max(1.0 - Sw, 1.0e-4)
    lambda_2 = vel * fg(Sw, nD, vel_mag) / ((phi_cell + eps) * sg)
    return lambda_1, lambda_2


def initial_condition():
    S = np.zeros((2, nx, ny), dtype=np.float64)
    Sw_0 = Sw_from_fw(fw_0, nd_0)
    S[0, :, :] = Sw_0
    S[1, :, :] = Sg(Sw_0) * nd_0
    return S


def apply_bc(S):
    S[0, 0, :] = Sw_inj
    S[1, 0, :] = Sg(Sw_inj) * nd_inj
    S[1, :, :] = np.maximum(S[1, :, :], 0.0)
    return S


def extract_nD(S):
    Sw = S[0]
    Sg_now = Sg(Sw)
    return np.where(Sg_now > 1.0e-12, S[1] / (Sg_now + eps), 0.0)


def foam_source(S, phi):
    src = np.zeros_like(S)
    Sw_now = S[0]
    Sg_now = Sg(Sw_now)
    nD_now = extract_nD(S)
    rate = Kc * nmax * (nd_le(Sw_now) - nD_now)
    src[1] = phi * Sg_now * rate / nmax
    return src


def KNP_flux_x(S, ux, phi):
    S_L, S_R = reconstruct_x(S)
    H = np.zeros((2, nx - 1, ny), dtype=np.float64)

    for i in range(nx - 1):
        for j in range(ny):
            Sw_L = S_L[0, i, j]
            Sw_R = S_R[0, i, j]
            Sg_L = Sg(Sw_L)
            Sg_R = Sg(Sw_R)
            nD_L = S_L[1, i, j] / Sg_L if Sg_L > 1.0e-12 else 0.0
            nD_R = S_R[1, i, j] / Sg_R if Sg_R > 1.0e-12 else 0.0
            u_face = ux[i + 1, j]
            phi_face = 0.5 * (phi[i, j] + phi[i + 1, j])
            l1_L, l2_L = eigvals(Sw_L, nD_L, u_face, phi_face)
            l1_R, l2_R = eigvals(Sw_R, nD_R, u_face, phi_face)
            a_plus = max(0.0, l1_L, l2_L, l1_R, l2_R)
            a_minus = min(0.0, l1_L, l2_L, l1_R, l2_R)
            hL = flux_x(Sw_L, nD_L, u_face)
            hR = flux_x(Sw_R, nD_R, u_face)
            sL = S_L[:, i, j]
            sR = S_R[:, i, j]
            H[:, i, j] = (a_plus * hL - a_minus * hR + a_plus * a_minus * (sR - sL)) / (a_plus - a_minus + eps)

    return H


def KNP_flux_y(S, uy, phi):
    if np.max(np.abs(uy)) <= 0.0:
        return np.zeros((2, nx, ny - 1), dtype=np.float64)

    S_L, S_R = reconstruct_y(S)
    H = np.zeros((2, nx, ny - 1), dtype=np.float64)

    for i in range(nx):
        for j in range(ny - 1):
            Sw_L = S_L[0, i, j]
            Sw_R = S_R[0, i, j]
            Sg_L = Sg(Sw_L)
            Sg_R = Sg(Sw_R)
            nD_L = S_L[1, i, j] / Sg_L if Sg_L > 1.0e-12 else 0.0
            nD_R = S_R[1, i, j] / Sg_R if Sg_R > 1.0e-12 else 0.0
            v_face = uy[i, j + 1]
            phi_face = 0.5 * (phi[i, j] + phi[i, j + 1])
            l1_L, l2_L = eigvals(Sw_L, nD_L, v_face, phi_face)
            l1_R, l2_R = eigvals(Sw_R, nD_R, v_face, phi_face)
            a_plus = max(0.0, l1_L, l2_L, l1_R, l2_R)
            a_minus = min(0.0, l1_L, l2_L, l1_R, l2_R)
            hL = flux_y(Sw_L, nD_L, v_face)
            hR = flux_y(Sw_R, nD_R, v_face)
            sL = S_L[:, i, j]
            sR = S_R[:, i, j]
            H[:, i, j] = (a_plus * hL - a_minus * hR + a_plus * a_minus * (sR - sL)) / (a_plus - a_minus + eps)

    return H


def compute_rhs(S, ux, uy, phi):
    Hx = KNP_flux_x(S, ux, phi)
    Hy = KNP_flux_y(S, uy, phi)
    rhs = np.zeros_like(S)

    f_in = np.zeros((2, ny))
    for j in range(ny):
        f_in[:, j] = flux_x(Sw_inj, nd_inj, ux[0, j])

    Sw_out = S[0, -1, :]
    nD_out = extract_nD(S)[-1, :]
    f_out = np.zeros((2, ny))
    for j in range(ny):
        f_out[:, j] = flux_x(Sw_out[j], nD_out[j], ux[-1, j])

    rhs[:, 0, :] -= (Hx[:, 0, :] - f_in) / dx
    rhs[:, 1:-1, :] -= (Hx[:, 1:, :] - Hx[:, :-1, :]) / dx
    rhs[:, -1, :] -= (f_out - Hx[:, -1, :]) / dx

    if ny > 1 and np.max(np.abs(uy)) > 0.0:
        nD_now = extract_nD(S)
        f_bottom = np.zeros((2, nx))
        f_top = np.zeros((2, nx))
        for i in range(nx):
            f_bottom[:, i] = flux_y(S[0, i, 0], nD_now[i, 0], uy[i, 0])
            f_top[:, i] = flux_y(S[0, i, -1], nD_now[i, -1], uy[i, -1])

        rhs[:, :, 0] -= (Hy[:, :, 0] - f_bottom) / dy
        if ny > 2:
            rhs[:, :, 1:-1] -= (Hy[:, :, 1:] - Hy[:, :, :-1]) / dy
        rhs[:, :, -1] -= (f_top - Hy[:, :, -1]) / dy

    rhs += foam_source(S, phi)
    return rhs / (phi[np.newaxis, :, :] + eps)


def ssprk3(S, dt, ux, uy, phi):
    k1 = compute_rhs(S, ux, uy, phi)
    S1 = apply_bc(S + dt * k1)
    k2 = compute_rhs(S1, ux, uy, phi)
    S2 = apply_bc(0.75 * S + 0.25 * (S1 + dt * k2))
    k3 = compute_rhs(S2, ux, uy, phi)
    return apply_bc((1.0 / 3.0) * S + (2.0 / 3.0) * (S2 + dt * k3))


def compute_dt(S, ux, uy, phi):
    Sw = S[0]
    Sg_now = 1.0 - Sw
    nD = extract_nD(S)
    h = 1.0e-6
    Swp = Sw + h
    Swm = Sw - h
    ux_cell = np.maximum(np.abs(ux[:-1, :]), np.abs(ux[1:, :]))
    uy_cell = np.maximum(np.abs(uy[:, :-1]), np.abs(uy[:, 1:]))
    dfw_x = (fw(Swp, nD, ux_cell) - fw(Swm, nD, ux_cell)) / (Swp - Swm + eps)
    dfw_y = (fw(Swp, nD, uy_cell) - fw(Swm, nD, uy_cell)) / (Swp - Swm + eps)
    ax = np.maximum(
        ux_cell * np.abs(dfw_x) / phi,
        ux_cell * np.abs(fg(Sw, nD, ux_cell)) / (phi * Sg_now + eps),
    )
    ay = np.maximum(
        uy_cell * np.abs(dfw_y) / phi,
        uy_cell * np.abs(fg(Sw, nD, uy_cell)) / (phi * Sg_now + eps),
    )
    dt_x = np.where(ax > 1.0e-30, dx / (ax + eps), 1.0e30)
    dt_y = np.where(ay > 1.0e-30, dy / (ay + eps), 1.0e30)
    return cfl * np.min(np.minimum(dt_x, dt_y))


def advance_transport(S, ux, uy, phi, t_start, t_stop):
    t = t_start
    step = 0
    while t < t_stop:
        dt = compute_dt(S, ux, uy, phi)
        if t + dt > t_stop:
            dt = t_stop - t
        S = ssprk3(S, dt, ux, uy, phi)
        t += dt
        step += 1
        print(f"Transport step {step}: t = {t:.2f}/{t_stop:.2f} s, dt = {dt:.2e} s")
    return S, t


def run_simulation():
    phi = np.full((nx, ny), phi_value)
    darcy_data = create_darcy_data()
    S = apply_bc(initial_condition())
    t = 0.0
    macro_step = 0
    ux = None
    uy = None
    pressure_array = None

    while t < t_final:
        _, ux, uy, pressure_array = solve_darcy(S[0], extract_nD(S), darcy_data)
        t_macro = min(t + dt_u, t_final)
        S, t = advance_transport(S, ux, uy, phi, t, t_macro)
        macro_step += 1
        print(f"Macro step {macro_step}: t = {t:.2f}/{t_final:.2f} s")

    save_results(S, ux, uy, pressure_array)


def write_scalar_values(file, values):
    for value in values.ravel(order="F"):
        file.write(f"{value:.16e}\n")


def write_vector_values(file, vx, vy):
    for vx_value, vy_value in zip(vx.ravel(order="F"), vy.ravel(order="F")):
        file.write(f"{vx_value:.16e} {vy_value:.16e} 0.0\n")


def save_vtk_results(S, ux=None, uy=None, pressure_array=None):
    vtk_path = output_dir / "resultado_final_caso2testes.vtk"

    with vtk_path.open("w") as file:

        file.write("SCALARS Sw")
        write_scalar_values(file, S[0])

        file.write("SCALARS pressure")
        write_scalar_values(file, pressure_array)

        ux_cell = 0.5 * (ux[:-1, :] + ux[1:, :])
        uy_cell = 0.5 * (uy[:, :-1] + uy[:, 1:])

        file.write("VECTORS velocity")
        write_vector_values(file, ux_cell, uy_cell)


def save_results(S, ux=None, uy=None, pressure_array=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    X, Y = create_coordinates()
    plt.figure(figsize=(12, 4))
    pcm = plt.pcolormesh(X, Y, S[0], cmap="jet", shading="auto", vmin=Swc, vmax=1.0)
    plt.colorbar(pcm, label="Water saturation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Water saturation in the porous medium")
    plt.tight_layout()
    plt.savefig(output_dir / "saturacao_final_caso2testes.png", dpi=200)
    plt.close()
    save_vtk_results(S, ux, uy, pressure_array)


run_simulation()

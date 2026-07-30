from pathlib import Path

import numpy as np

from RT_FEM import Lx, Ly, nx, ny, u_inj, create_darcy_data, solve_darcy


muw = 1.0e-3
mug = 2.0e-5

Swc = 0.2
Sgr = 0.0
Sstar = 0.37
Sw0 = 0.9
Sw_inj = 0.372

nd_0 = 0.0
nd_inj = 0.0
nmax = 8.0e13

A_foam = 400.0
Kc = 1.0e-6
alpha_foam = 7.0e-16
foam_exponent = 2.0 / 3.0

phi_value = 0.2
dx = Lx / nx
dy = Ly / ny

t_final = 2000.0
dt_u = 10.0
dt_max = None
print_every = 1

theta = 1.0
cfl = 0.375
eps = 1.0e-12

output_dir = Path("output")
snapshot_file = Path("snap.txt")
snapshot_interval = 50.0
next_snapshot_interval = snapshot_interval


def Sg(Sw):
    return 1.0 - Sw


def extract_nD(S):
    Sg_now = Sg(S[0])
    return np.where(Sg_now > 1.0e-12, S[1] / (Sg_now + eps), 0.0)


def nd_le(Sw):
    return np.where(Sw < Sstar, 0.0, np.tanh(A_foam * (Sw - Sstar)))


def krw(Sw):
    Se = (Sw - Swc) / (1.0 - Swc - Sgr)
    return np.maximum(0.0, Se) ** 4


def krg(Sw):
    Sge = (1.0 - Sw - Sgr) / (1.0 - Swc - Sgr)
    return np.maximum(0.0, Sge) ** 2


def lambda_w(Sw):
    return krw(Sw) / muw


def foam_viscosity(Sw, nD, u=u_inj):
    if isinstance(Sw, np.ndarray):
        Sw, nD, u = np.broadcast_arrays(Sw, nD, np.abs(u))
        mu, sg, krg_val, lw, nf = np.full_like(Sw, mug), np.maximum(1-Sw, 0), krg(Sw), lambda_w(Sw), np.maximum(nD, 0)*nmax
        act = (nf > 0) & (sg > eps)
        
        if np.any(act):
            a = alpha_foam * nf[act] * lw[act] * (phi_value * sg[act] / (krg_val[act] * u[act] + eps))**foam_exponent / 3.0
            b = (krg_val[act] + lw[act] * mug) / 2.0
            sq = np.sqrt(2.0 * a**3 * b + b**2)
            mu[act] = mug + 3.0 * a * (a + np.cbrt(a**3 + b + sq) + np.cbrt(a**3 + b - sq))**2 / lw[act]
        return np.maximum(mu, mug)

    u_abs, sg, nf = abs(u), max(1.0 - Sw, 0.0), max(nD, 0.0) * nmax
    krg_val, lw = krg_s(Sw), lambda_w_s(Sw)
    
    if (nf > 0) and (sg > eps):
        a = alpha_foam * nf * lw * (phi_value * sg / (krg_val * u_abs))**foam_exponent / 3.0
        b = (krg_val + lw * mug) / 2.0
        sq = np.sqrt(2.0 * a**3 * b + b**2)
        root = a + np.cbrt(a**3 + b + sq) + np.cbrt(a**3 + b - sq)
        return max(mug + 3.0 * a * root**2 / lw , mug)
        
    return mug

    
def fw(Sw, nD, u=u_inj):
    lw = lambda_w(Sw)
    lg = krg(Sw) / foam_viscosity(Sw, nD, u)
    return lw / (lw + lg)


def fg(Sw, nD, u=u_inj):
    return 1.0 - fw(Sw, nD, u)


def flux(Sw, nD, vel):
    fw_value = fw(Sw, nD, np.abs(vel))
    return np.array([vel * fw_value, vel * nD * (1.0 - fw_value)])


def krw_s(Sw):
    Se = (Sw - Swc) / (1.0 - Swc - Sgr)
    if Se < 0.0:
        Se = 0.0
    return Se ** 4


def krg_s(Sw):
    Sge = (1.0 - Sw - Sgr) / (1.0 - Swc - Sgr)
    if Sge < 0.0:
        Sge = 0.0
    return Sge ** 2


def lambda_w_s(Sw):
    return krw_s(Sw) / muw


def fw_s(Sw, nD, u):
    lw = lambda_w_s(Sw)
    lg = krg_s(Sw) / foam_viscosity(Sw, nD, u)
    return lw / (lw + lg)


def minmod(a, b, c):
    out = np.zeros_like(a)
    pos = (a > 0.0) & (b > 0.0) & (c > 0.0)
    neg = (a < 0.0) & (b < 0.0) & (c < 0.0)
    out[pos] = np.minimum(np.minimum(a[pos], b[pos]), c[pos])
    out[neg] = np.maximum(np.maximum(a[neg], b[neg]), c[neg])
    return out


def reconstruct(S, axis):
    h = dx if axis == 1 else dy
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (1, 1)
    S_pad = np.pad(S, pad, mode="edge")

    sl = [slice(None)] * 3
    sc = [slice(None)] * 3
    sr = [slice(None)] * 3
    sl[axis] = slice(None, -2)
    sc[axis] = slice(1, -1)
    sr[axis] = slice(2, None)

    dL = (S_pad[tuple(sc)] - S_pad[tuple(sl)]) / h
    dR = (S_pad[tuple(sr)] - S_pad[tuple(sc)]) / h
    dC = (S_pad[tuple(sr)] - S_pad[tuple(sl)]) / (2.0 * h)
    slope = minmod(theta * dL, dC, theta * dR)

    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)

    return S[tuple(left)] + 0.5 * h * slope[tuple(left)], S[tuple(right)] - 0.5 * h * slope[tuple(right)]


def state_flux_eigs(Sw, nD, vel, phi_face):
    vel_abs = abs(vel)
    fw_value = fw_s(Sw, nD, vel_abs)
    fg_value = 1.0 - fw_value

    h = 1.0e-7
    dfw = (fw_s(Sw + h, nD, vel_abs) - fw_s(Sw - h, nD, vel_abs)) / (2.0 * h)

    sg = 1.0 - Sw
    if sg < 1.0e-10:
        sg = 1.0e-10

    l1 = vel * dfw / phi_face
    l2 = vel * fg_value / (sg * phi_face)

    return l1, l2, vel * fw_value, vel * nD * fg_value


def KNP_flux_x(S, ux, phi):
    S_L, S_R = reconstruct(S, axis=1)
    H = np.zeros((2, nx - 1, ny))

    for i in range(nx - 1):
        for j in range(ny):
            Sw_L = (S_L[0, i, j])
            Sw_R = (S_R[0, i, j])
            S1_L = (S_L[1, i, j])
            S1_R = (S_R[1, i, j])

            Sg_L = 1.0 - Sw_L
            Sg_R = 1.0 - Sw_R

            nD_L = S1_L / (Sg_L) if Sg_L > 1.0e-12 else 0.0
            nD_R = S1_R / (Sg_R) if Sg_R > 1.0e-12 else 0.0

            vel = (ux[i + 1, j])
            phi_face = 0.5 * ((phi[i, j]) + (phi[i + 1, j]))

            l1_L, l2_L, fL0, fL1 = state_flux_eigs(Sw_L, nD_L, vel, phi_face)
            l1_R, l2_R, fR0, fR1 = state_flux_eigs(Sw_R, nD_R, vel, phi_face)

            ap = max(0.0, l1_L, l2_L, l1_R, l2_R)
            am = min(0.0, l1_L, l2_L, l1_R, l2_R)

            den = ap - am + eps
            diff = ap * am

            H[0, i, j] = (ap * fL0 - am * fR0 + diff * (Sw_R - Sw_L)) / den
            H[1, i, j] = (ap * fL1 - am * fR1 + diff * (S1_R - S1_L)) / den

    return H


def KNP_flux_y(S, uy, phi):
    if np.max(np.abs(uy)) == 0.0:
        return np.zeros((2, nx, ny - 1))

    S_L, S_R = reconstruct(S, axis=2)
    H = np.zeros((2, nx, ny - 1))

    for i in range(nx):
        for j in range(ny - 1):
            Sw_L = (S_L[0, i, j])
            Sw_R = (S_R[0, i, j])
            S1_L = (S_L[1, i, j])
            S1_R = (S_R[1, i, j])

            Sg_L = 1.0 - Sw_L
            Sg_R = 1.0 - Sw_R

            nD_L = S1_L / (Sg_L) if Sg_L > 1.0e-12 else 0.0
            nD_R = S1_R / (Sg_R) if Sg_R > 1.0e-12 else 0.0

            vel = (uy[i, j + 1])
            phi_face = 0.5 * ((phi[i, j]) + (phi[i, j + 1]))

            l1_L, l2_L, fL0, fL1 = state_flux_eigs(Sw_L, nD_L, vel, phi_face)
            l1_R, l2_R, fR0, fR1 = state_flux_eigs(Sw_R, nD_R, vel, phi_face)

            ap = max(0.0, l1_L, l2_L, l1_R, l2_R)
            am = min(0.0, l1_L, l2_L, l1_R, l2_R)

            den = ap - am + eps
            diff = ap * am

            H[0, i, j] = (ap * fL0 - am * fR0 + diff * (Sw_R - Sw_L)) / den
            H[1, i, j] = (ap * fL1 - am * fR1 + diff * (S1_R - S1_L)) / den

    return H


def left_face_flux(S, ux, phi):
    S_L, S_R = reconstruct(S, axis=1)
    i = 0
    q = np.zeros(ny)

    for j in range(ny):
        Sw_L = (S_L[0, i, j])
        Sw_R = (S_R[0, i, j])
        S1_L = (S_L[1, i, j])
        S1_R = (S_R[1, i, j])

        Sg_L = 1.0 - Sw_L
        Sg_R = 1.0 - Sw_R

        nD_L = S1_L / (Sg_L) if Sg_L > 1.0e-12 else 0.0
        nD_R = S1_R / (Sg_R) if Sg_R > 1.0e-12 else 0.0

        vel = (ux[i + 1, j])
        phi_face = 0.5 * ((phi[i, j]) + (phi[i + 1, j]))

        l1_L, l2_L, fL0, fL1 = state_flux_eigs(Sw_L, nD_L, vel, phi_face)
        l1_R, l2_R, fR0, fR1 = state_flux_eigs(Sw_R, nD_R, vel, phi_face)

        ap = max(0.0, l1_L, l2_L, l1_R, l2_R)
        am = min(0.0, l1_L, l2_L, l1_R, l2_R)

        den = ap - am + eps
        diff = ap * am

        q[j] = (ap * fL0 - am * fR0 + diff * (Sw_R - Sw_L)) / den

    return q


def foam_source(S, phi):
    Sw = S[0]
    src = np.zeros_like(S)
    src[1] = phi * Sg(Sw) * Kc * (nd_le(Sw) - extract_nD(S))
    return src


def compute_rhs(S, ux, uy, phi):
    Hx = KNP_flux_x(S, ux, phi)
    Hy = KNP_flux_y(S, uy, phi)
    rhs = np.zeros_like(S)
    nD = extract_nD(S)

    f_out = flux(S[0, -1, :], nD[-1, :], ux[-1, :])

    rhs[:, 1:-1, :] -= (Hx[:, 1:, :] - Hx[:, :-1, :]) / dx

    rhs[:, -1, :] -= (f_out - Hx[:, -1, :]) / dx

    if ny > 1 and np.max(np.abs(uy)) > 0.0:
        f_bottom = flux(S[0, :, 0], nD[:, 0], uy[:, 0])
        f_top = flux(S[0, :, -1], nD[:, -1], uy[:, -1])

        rhs[:, :, 0] -= (Hy[:, :, 0] - f_bottom) / dy
        rhs[:, :, 1:-1] -= (Hy[:, :, 1:] - Hy[:, :, :-1]) / dy
        rhs[:, :, -1] -= (f_top - Hy[:, :, -1]) / dy

    rhs += foam_source(S, phi)
    rhs[:, 0, :] = 0.0

    return rhs / (phi[np.newaxis, :, :])

def apply_bc(S):
    S[0, 0, :] = Sw_inj
    S[1, 0, :] = Sg(Sw_inj) * nd_inj
    return S


def ssprk3(S, dt, ux, uy, phi):
    k1 = compute_rhs(S, ux, uy, phi)
    S1 = apply_bc(S + dt * k1)
    k2 = compute_rhs(S1, ux, uy, phi)
    S2 = apply_bc(0.75 * S + 0.25 * (S1 + dt * k2))
    k3 = compute_rhs(S2, ux, uy, phi)
    return apply_bc((S + 2.0 * (S2 + dt * k3)) / 3.0)


def compute_dt(S, ux, uy, phi):
    Sw = S[0]
    nD = extract_nD(S)
    Sg_now = np.maximum(Sg(Sw), 1.0e-10)

    ux_cell = np.maximum(np.abs(ux[:-1, :]), np.abs(ux[1:, :]))
    uy_cell = np.maximum(np.abs(uy[:, :-1]), np.abs(uy[:, 1:]))

    h = 1.0e-6
    dfw_x = (fw(Sw + h, nD, ux_cell) - fw(Sw - h, nD, ux_cell)) / (2.0 * h)
    dfw_y = (fw(Sw + h, nD, uy_cell) - fw(Sw - h, nD, uy_cell)) / (2.0 * h)

    ax = np.maximum(
        ux_cell * np.abs(dfw_x) / phi,
        ux_cell * np.abs(fg(Sw, nD, ux_cell)) / (phi * Sg_now),
    )
    ay = np.maximum(
        uy_cell * np.abs(dfw_y) / phi,
        uy_cell * np.abs(fg(Sw, nD, uy_cell)) / (phi * Sg_now),
    )

    amax_x = np.max(ax)
    amax_y = np.max(ay)
    dt = cfl / (amax_x / dx + amax_y / dy)
    return min(dt, dt_max) if dt_max is not None else dt


def initial_condition():
    S = np.zeros((2, nx, ny))
    S[0] = Sw0
    S[1] = Sg(Sw0) * nd_0
    return apply_bc(S)


def water_mass_active_domain(S, phi):
    return np.sum(phi[1:, :] * S[0, 1:, :]) * dx * dy


def water_boundary(S, ux, uy, phi):
    f_left = left_face_flux(S, ux, phi)
    nD = extract_nD(S)

    f_right = flux(S[0, -1, :], nD[-1, :], ux[-1, :])[0]

    q_in = np.sum(f_left) * dy
    q_out = np.sum(f_right) * dy

    return q_in, q_out

def advance_transport(S, ux, uy, phi, pressure, t_start, t_stop):
    t = t_start
    step = 0
    water_in = 0.0
    water_out = 0.0

    while t < t_stop:
        target = t_stop
        dt = min(compute_dt(S, ux, uy, phi), target - t)

        q_in, q_out = water_boundary(S, ux, uy, phi)
        water_in += q_in * dt
        water_out += q_out * dt

        S = ssprk3(S, dt, ux, uy, phi)
        t += dt
        step += 1

        print(
            f"Transport step {step}: t={t:.2f}/{t_stop:.2f}, "
            f"dt={dt:.3e}, Sw=[{S[0].min():.4f}, {S[0].max():.4f}]"
        )

        if step % 50 == 0 or t >= t_stop:
            save_snapshot_txt(S, ux, uy, pressure, t)

    return S, t, water_in, water_out


def save_snapshot_file(S, ux, uy, pressure, t, tag="snapshot"):
    path = output_dir / "snapshots" / f"{tag}_t_{t:.2f}.vti"
    save_vti(S, ux, uy, pressure, path)
    print(f"Snapshot salvo: {path}")


def save_snapshot_txt(S, ux, uy, pressure, t):
    global next_snapshot_interval

    if snapshot_file.exists():
        save_snapshot_file(S, ux, uy, pressure, t, tag="manual")
        snapshot_file.unlink(missing_ok=True)

    if snapshot_interval is not None and t >= next_snapshot_interval:
        save_snapshot_file(S, ux, uy, pressure, next_snapshot_interval, tag="auto")
        next_snapshot_interval += snapshot_interval


def save_results(S, ux, uy, pressure):
    save_vti(S, ux, uy, pressure, output_dir / "resultado_caso2testes.vti")


def save_vti(S, ux, uy, pressure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Sw = S[0]
    nD = extract_nD(S)
    ux_cell = 0.5 * (ux[:-1, :] + ux[1:, :])
    uy_cell = 0.5 * (uy[:, :-1] + uy[:, 1:])

    def array_text(values):
        return " ".join(f"{v:.8e}" for v in values.ravel(order="F"))

    velocity = np.array([ux_cell, uy_cell, np.zeros_like(ux_cell)])
    velocity_text = " ".join(f"{vx:.8e} {vy:.8e} {vz:.8e}" for vx, vy, vz in velocity.reshape(3, -1, order="F").T)

    path.write_text(f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {nx - 1} 0 {ny - 1} 0 0" Origin="{0.5 * dx} {0.5 * dy} 0" Spacing="{dx} {dy} 1">
    <Piece Extent="0 {nx - 1} 0 {ny - 1} 0 0">
      <PointData Scalars="Sw" Vectors="velocity">
        <DataArray type="64" Name="Sw" format="ascii">{array_text(Sw)}</DataArray>
        <DataArray type="64" Name="nD" format="ascii">{array_text(nD)}</DataArray>
        <DataArray type="64" Name="pressure" format="ascii">{array_text(pressure)}</DataArray>
        <DataArray type="64" Name="velocity" NumberOfComponents="3" format="ascii">{velocity_text}</DataArray>
      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
''')


def run_simulation():
    phi = np.full((nx, ny), phi_value)
    darcy_data = create_darcy_data()

    S = initial_condition()
    t = 0.0
    ux = uy = pressure = None

    while t < t_final:
        _, ux, uy, pressure = solve_darcy(S[0], extract_nD(S), darcy_data)

        mass_before = water_mass_active_domain(S, phi)

        S, t, water_in, water_out = advance_transport(
            S, ux, uy, phi, pressure, t, min(t + dt_u, t_final)
        )

        mass_after = water_mass_active_domain(S, phi)
        delta_mass = mass_after - mass_before

        print(
            f"Macro step: t={t:.2f}/{t_final:.2f} | "
            f"agua entrou={water_in:.6e}, "
            f"agua saiu={water_out:.6e}, "
            f"delta massa={delta_mass:.6e}, "
        )

    save_results(S, ux, uy, pressure)


if __name__ == "__main__":
    run_simulation()
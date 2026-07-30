from pathlib import Path

import numpy as np
import ufl
from basix.ufl import element
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import default_real_type, fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem


muw = 1.0e-3
mug = 2.0e-5
Swc = 0.2
Sgr = 0.0
u_inj = 3.0e-5
nmax = 8.0e13
alpha_foam = 7.0e-16
foam_exponent = 2.0 / 3.0
phi_value = 0.2

Lx = 3.67
Ly = 1.0
nx = 220
ny = 60
dx = Lx / nx
dy = Ly / ny

permeability_file = "spe10_layer_36.pbt"
eps = 1.0e-12


def load_permeability():
    data = Path(permeability_file).read_text().splitlines()
    perm_raw = np.array([float(value) for value in data]).reshape(ny, nx).T
    return perm_raw * 9.869233e-16


def Sg(Sw):
    return 1.0 - Sw


def krw(Sw):
    Se = (Sw - Swc) / (1.0 - Swc - Sgr + eps)
    return np.maximum(0.0, Se) ** 4


def krg(Sw):
    Sge = (1.0 - Sw - Sgr) / (1.0 - Swc - Sgr + eps)
    return np.maximum(0.0, Sge) ** 2


def lambda_w(Sw):
    return krw(Sw) / muw


def foam_viscosity(Sw, nD, u=u_inj):
    Sw, nD, u = np.broadcast_arrays(Sw, nD, np.abs(u))

    mu = np.full_like(Sw, mug)
    sg = np.maximum(Sg(Sw), 0.0)
    krg_value = krg(Sw)
    lw = lambda_w(Sw)
    nf = np.maximum(nD, 0.0) * nmax

    active = (nf > 0.0) & (u > eps) & (sg > eps) & (krg_value > eps) & (lw > eps)
    if not np.any(active):
        return mu

    beta = phi_value * sg[active] / (krg_value[active] * u[active] + eps)
    a = alpha_foam * nf[active] * lw[active] * beta**foam_exponent / 3.0
    b = (krg_value[active] + lw[active] * mug) / 2.0
    a3 = a**3

    sqrt_term = np.sqrt(2.0 * a3 * b + b**2)
    root = a + np.cbrt(a3 + b + sqrt_term) + np.cbrt(a3 + b - sqrt_term)

    mu[active] = mug + 3.0 * a * root**2 / (lw[active] + eps)
    return np.maximum(mu, mug)


def lambda_g(Sw, nD, u=u_inj):
    return krg(Sw) / foam_viscosity(Sw, nD, u)


def lambda_total(Sw, nD, u=u_inj):
    return lambda_w(Sw) + lambda_g(Sw, nD, u)


def create_rt_element(msh):
    return element("RT", msh.basix_cell(), 1, dtype=default_real_type)


def create_pressure_element(msh):
    return element("DG", msh.basix_cell(), 0, dtype=default_real_type)


def compute_point_cells(msh, points):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    return np.array(
        [colliding.links(i)[0] for i in range(points.shape[0])],
    )


def make_points(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.zeros((X.size, 3), dtype=default_real_type)
    points[:, 0] = X.ravel()
    points[:, 1] = Y.ravel()
    return points


def create_face_evaluation_data(msh):
    tol = 1.0e-12

    x_faces = np.linspace(0.0, Lx, nx + 1)
    y_centers = (np.arange(ny) + 0.5) * dy

    x_face_points = make_points(x_faces, y_centers)

    x_locator = x_faces.copy()
    x_locator[0] += tol
    x_locator[1:] -= tol

    x_face_locator_points = make_points(x_locator, y_centers)

    x_centers = (np.arange(nx) + 0.5) * dx
    y_faces = np.linspace(0.0, Ly, ny + 1)

    y_face_points = make_points(x_centers, y_faces)

    y_locator = y_faces.copy()
    y_locator[0] += tol
    y_locator[1:] -= tol

    y_face_locator_points = make_points(x_centers, y_locator)

    return {
        "x_face_points": x_face_points,
        "x_face_cells": compute_point_cells(msh, x_face_locator_points),
        "y_face_points": y_face_points,
        "y_face_cells": compute_point_cells(msh, y_face_locator_points),
    }


def extract_face_velocities(rt_flux, darcy_data):
    x_values = rt_flux.eval(
        darcy_data["x_face_points"],
        darcy_data["x_face_cells"],
    )

    y_values = rt_flux.eval(
        darcy_data["y_face_points"],
        darcy_data["y_face_cells"],
    )

    ux_face = x_values[:, 0].reshape(nx + 1, ny)
    uy_face = y_values[:, 1].reshape(nx, ny + 1)

    ux_face[0, :] = u_inj
    uy_face[:, 0] = 0.0
    uy_face[:, -1] = 0.0

    return ux_face, uy_face


def build_face_tags(msh):
    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], Lx))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], Ly))

    facet_indices = np.hstack([left_facets, right_facets, bottom_facets, top_facets])
    facet_markers = np.hstack(
        [
            np.full_like(left_facets, 1),
            np.full_like(right_facets, 2),
            np.full_like(bottom_facets, 3),
            np.full_like(top_facets, 4),
        ]
    )

    order = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(msh, fdim, facet_indices[order], facet_markers[order])

    return {
        "fdim": fdim,
        "left_facets": left_facets,
        "right_facets": right_facets,
        "bottom_facets": bottom_facets,
        "top_facets": top_facets,
        "facet_tag": facet_tag,
    }


def create_darcy_data():
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [Lx, Ly]],
        [nx, ny],
        cell_type=mesh.CellType.quadrilateral,
    )

    face_tags = build_face_tags(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=face_tags["facet_tag"])

    U = fem.functionspace(msh, create_rt_element(msh))
    Q = fem.functionspace(msh, create_pressure_element(msh))

    coords = Q.tabulate_dof_coordinates()
    ii = np.minimum(np.floor(coords[:, 0] / dx).astype(int), nx - 1)
    jj = np.minimum(np.floor(coords[:, 1] / dy).astype(int), ny - 1)

    K = fem.Function(Q, name="Permeability")
    T = fem.Function(Q, name="TotalMobilityTimesK")
    K.x.array[:] = load_permeability()[ii, jj]

    left_cells = mesh.compute_incident_entities(
        msh.topology,
        face_tags["left_facets"],
        face_tags["fdim"],
        msh.topology.dim,
    )

    left_dofs = fem.locate_dofs_topological(U, face_tags["fdim"], face_tags["left_facets"])
    top_dofs = fem.locate_dofs_topological(U, face_tags["fdim"], face_tags["top_facets"])
    bottom_dofs = fem.locate_dofs_topological(U, face_tags["fdim"], face_tags["bottom_facets"])

    left_flux = fem.Function(U)
    zero_flux = fem.Function(U)

    left_flux.interpolate(
        lambda x: np.vstack((u_inj * np.ones_like(x[0]), np.zeros_like(x[0]))),
        cells0=left_cells,
    )
    zero_flux.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]))))

    return {
        "msh": msh,
        "U": U,
        "Q": Q,
        "K": K,
        "T": T,
        "ds": ds,
        "ii": ii,
        "jj": jj,
        "bcs": [
            fem.dirichletbc(left_flux, left_dofs),
            fem.dirichletbc(zero_flux, top_dofs),
            fem.dirichletbc(zero_flux, bottom_dofs),
        ],
        **create_face_evaluation_data(msh),
    }


def solve_darcy(Sw, nD, darcy_data):
    U_space = darcy_data["U"]
    Q_space = darcy_data["Q"]
    K = darcy_data["K"]
    T = darcy_data["T"]
    ds = darcy_data["ds"]
    ii = darcy_data["ii"]
    jj = darcy_data["jj"]

    T.x.array[:] = K.x.array * lambda_total(Sw, nD, u_inj)[ii, jj]

    W = ufl.MixedFunctionSpace(U_space, Q_space)
    flux, pressure = ufl.TrialFunctions(W)
    tau, v = ufl.TestFunctions(W)
    zero = fem.Constant(darcy_data["msh"], ScalarType(0.0))

    a = ufl.extract_blocks(
        (1.0 / T) * ufl.inner(flux, tau) * ufl.dx
        - pressure * ufl.div(tau) * ufl.dx
        + ufl.div(flux) * v * ufl.dx
        + zero * pressure * v * ufl.dx
    )
    L = [
        zero * ufl.dot(tau, ufl.FacetNormal(darcy_data["msh"])) * ds(2),
        ufl.ZeroBaseForm((v,)),
    ]

    U_h = fem.Function(U_space, name="Flux")
    pressure_h = fem.Function(Q_space, name="Pressure")

    problem = LinearProblem(
        a,
        L,
        u=[U_h, pressure_h],
        kind="mpi",
        bcs=darcy_data["bcs"],
        petsc_options_prefix="mixed_rt_darcy_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    problem.solve()
    

    ux_face, uy_face = extract_face_velocities(U_h, darcy_data)

    pressure_array = np.zeros((nx, ny))
    pressure_array[ii, jj] = pressure_h.x.array

    return pressure_h, ux_face, uy_face, pressure_array
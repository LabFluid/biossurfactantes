from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem


# Parametros
muw = 1.0e-3
mug = 2.0e-5
Swc = 0.2
Sgr = 0.0
u_inj = 3.0e-5

# Malha
Lx = 3.67
Ly = 1.0
nx = 220
ny = 60
dx = Lx / nx
dy = Ly / ny

# Permeabilidade
permeability_file = "spe10_layer_36.pbt"
eps = 1.0e-12


def load_permeability():
    data = Path(permeability_file).read_text().splitlines()
    perm_raw = np.array([float(value) for value in data], dtype=np.float64).reshape(ny, nx).T
    return perm_raw * 9.869233e-16


def krw(Sw):
    Se = (Sw - Swc) / (1.0 - Swc - Sgr + eps)
    return np.maximum(0.0, Se) ** 4


def krg0(Sw):
    Sge = (1.0 - Sw - Sgr) / (1.0 - Swc - Sgr + eps)
    return np.maximum(0.0, Sge) ** 2


def krg(Sw, nD):
    return krg0(Sw) / (18500.0 * nD + 1.0)


def lambda_w(Sw):
    return krw(Sw) / muw


def lambda_g(Sw, nD):
    return krg(Sw, nD) / mug


def lambda_total(Sw, nD):
    return lambda_w(Sw) + lambda_g(Sw, nD)


def create_darcy_data():
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [Lx, Ly]],
        [nx, ny],
        cell_type=mesh.CellType.quadrilateral,
    )
    tdim = msh.topology.dim
    fdim = tdim - 1

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
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    V = fem.functionspace(msh, ("CG", 1))
    V0 = fem.functionspace(msh, ("DG", 0))

    coords0 = V0.tabulate_dof_coordinates()
    print(coords0)
    ii = np.floor(coords0[:, 0] / dx).astype(int)
    print(ii)
    jj = np.floor(coords0[:, 1] / dy).astype(int)

    K = fem.Function(V0)
    K.name = "Permeability"
    T = fem.Function(V0)
    T.name = "TotalMobilityTimesK"

    permeability = load_permeability()
    K.x.array[:] = permeability[ii, jj]

    dofs_right = fem.locate_dofs_topological(V, fdim, right_facets)
    bc_right = fem.dirichletbc(ScalarType(0.0), dofs_right, V)
    ubar_left = fem.Constant(msh, ScalarType(-u_inj))

    return {
        "msh": msh,
        "V": V,
        "V0": V0,
        "K": K,
        "T": T,
        "ds": ds,
        "bc_right": bc_right,
        "ubar_left": ubar_left,
        "ii": ii,
        "jj": jj,
        "permeability": permeability,
    }


def solve_darcy(Sw, nD, darcy_data):
    T = darcy_data["T"]
    K = darcy_data["K"]
    V = darcy_data["V"]
    V0 = darcy_data["V0"]
    ds = darcy_data["ds"]
    bc_right = darcy_data["bc_right"]
    ubar_left = darcy_data["ubar_left"]
    ii = darcy_data["ii"]
    jj = darcy_data["jj"]

    lambda_t = lambda_total(Sw, nD)
    T.x.array[:] = K.x.array * lambda_t[ii, jj]

    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(T * ufl.grad(p), ufl.grad(v)) * ufl.dx
    L = -ubar_left * v * ds(1)

    problem = LinearProblem(
        a,
        L,
        bcs=[bc_right],
        petsc_options_prefix="darcy_coupled_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    pressure = problem.solve()
    pressure.name = "Pressure"

    ux_fun = fem.Function(V0, name="ux")
    uy_fun = fem.Function(V0, name="uy")

    expr_ux = fem.Expression((-T * ufl.grad(pressure))[0], V0.element.interpolation_points)
    expr_uy = fem.Expression((-T * ufl.grad(pressure))[1], V0.element.interpolation_points)

    ux_fun.interpolate(expr_ux)
    uy_fun.interpolate(expr_uy)

    ux = np.zeros((nx, ny), dtype=np.float64)
    uy = np.zeros((nx, ny), dtype=np.float64)
    ux[ii, jj] = ux_fun.x.array
    uy[ii, jj] = uy_fun.x.array

    return pressure, ux, uy

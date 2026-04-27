from pathlib import Path
from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem


# Parametros
muw = 1.0e-3
mug = 2.0e-5
Swc = 0.2
Sgr = 0.0
u_inj = 3.0e-5
pressure_right = 0.0

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


def create_rt_element(msh):
    try:
        return element("RT", msh.basix_cell(), 1, dtype=default_real_type)
    except ValueError:
        return element("RTCF", msh.basix_cell(), 1, dtype=default_real_type)


def create_pressure_element(msh):
    try:
        return element("Discontinuous Lagrange", msh.basix_cell(), 0, dtype=default_real_type)
    except ValueError:
        return element("DQ", msh.basix_cell(), 0, dtype=default_real_type)


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

    dx_measure = ufl.Measure("dx", domain=msh)
    ds_measure = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    rt_element = create_rt_element(msh)
    pressure_element = create_pressure_element(msh)

    V = fem.functionspace(msh, rt_element)
    Q = fem.functionspace(msh, pressure_element)
    V0 = fem.functionspace(msh, ("DG", 0))
    system_space = ufl.MixedFunctionSpace(V, Q)

    coords0 = V0.tabulate_dof_coordinates()
    ii = np.floor(coords0[:, 0] / dx).astype(int)
    jj = np.floor(coords0[:, 1] / dy).astype(int)

    permeability = load_permeability()

    K = fem.Function(V0, name="Permeability")
    T = fem.Function(V0, name="TotalMobilityTimesK")
    K.x.array[:] = permeability[ii, jj]
    pressure_right_value = fem.Constant(msh, default_real_type(pressure_right))

    dofs_left = fem.locate_dofs_topological(V, fdim, left_facets)
    dofs_top = fem.locate_dofs_topological(V, fdim, top_facets)
    dofs_bottom = fem.locate_dofs_topological(V, fdim, bottom_facets)

    cells_left = mesh.compute_incident_entities(msh.topology, left_facets, fdim, tdim)

    prescribed_flux = fem.Function(V, name="PrescribedFlux")
    prescribed_flux.x.array[:] = 0.0
    prescribed_flux.interpolate(
        lambda x: np.vstack((u_inj * np.ones_like(x[0]), np.zeros_like(x[0]))),
        cells0=cells_left,
    )

    bc_left = fem.dirichletbc(prescribed_flux, dofs_left)
    bc_top = fem.dirichletbc(prescribed_flux, dofs_top)
    bc_bottom = fem.dirichletbc(prescribed_flux, dofs_bottom)

    return {
        "msh": msh,
        "V": V,
        "Q": Q,
        "V0": V0,
        "system_space": system_space,
        "K": K,
        "T": T,
        "pressure_right_value": pressure_right_value,
        "dx": dx_measure,
        "ds": ds_measure,
        "facet_tag": facet_tag,
        "left_facets": left_facets,
        "right_facets": right_facets,
        "top_facets": top_facets,
        "bottom_facets": bottom_facets,
        "bcs": [bc_left, bc_top, bc_bottom],
        "ii": ii,
        "jj": jj,
        "permeability": permeability,
    }


def compute_boundary_fluxes(flux, darcy_data):
    msh = darcy_data["msh"]
    ds_measure = darcy_data["ds"]
    normal = ufl.FacetNormal(msh)

    fluxes = {}
    for marker, name in [(1, "left"), (2, "right"), (3, "bottom"), (4, "top")]:
        form = fem.form(ufl.dot(flux, normal) * ds_measure(marker))
        value = fem.assemble_scalar(form)
        fluxes[name] = msh.comm.allreduce(value, op=MPI.SUM)

    fluxes["mass_in"] = -fluxes["left"]
    fluxes["mass_out"] = fluxes["right"] + fluxes["top"] + fluxes["bottom"]
    fluxes["net_outward"] = fluxes["left"] + fluxes["right"] + fluxes["top"] + fluxes["bottom"]
    fluxes["imbalance"] = fluxes["mass_in"] - fluxes["mass_out"]
    return fluxes


def solve_darcy(Sw, nD, darcy_data, return_diagnostics=False):
    V = darcy_data["V"]
    Q = darcy_data["Q"]
    V0 = darcy_data["V0"]
    system_space = darcy_data["system_space"]
    K = darcy_data["K"]
    T = darcy_data["T"]
    pressure_right_value = darcy_data["pressure_right_value"]
    dx_measure = darcy_data["dx"]
    ds_measure = darcy_data["ds"]
    bcs = darcy_data["bcs"]
    ii = darcy_data["ii"]
    jj = darcy_data["jj"]

    lambda_t = lambda_total(Sw, nD)
    T.x.array[:] = K.x.array * lambda_t[ii, jj]

    sigma, pressure = ufl.TrialFunctions(system_space)
    tau, w = ufl.TestFunctions(system_space)
    normal = ufl.FacetNormal(darcy_data["msh"])

    inv_T = 1.0 / T
    mixed_form = (
        inv_T * ufl.inner(sigma, tau) * dx_measure
        - pressure * ufl.div(tau) * dx_measure
        + ufl.div(sigma) * w * dx_measure
    )
    a = ufl.extract_blocks(mixed_form)
    L = [
        -pressure_right_value * ufl.dot(tau, normal) * ds_measure(2),
        ufl.ZeroBaseForm((w,)),
    ]
    a_p = ufl.extract_blocks(
        inv_T * ufl.inner(sigma, tau) * dx_measure
        + ufl.inner(ufl.div(sigma), ufl.div(tau)) * dx_measure
        + pressure * w * dx_measure
    )

    flux_solution = fem.Function(V, name="Flux")
    pressure_solution = fem.Function(Q, name="Pressure")

    problem = LinearProblem(
        a,
        L,
        u=[flux_solution, pressure_solution],
        P=a_p,
        kind="nest",
        bcs=bcs,
        petsc_options_prefix="rt_mixed_darcy_",
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "ksp_rtol": 1.0e-10,
            "ksp_gmres_restart": 100,
        },
    )

    problem.solve()

    ux_fun = fem.Function(V0, name="ux")
    uy_fun = fem.Function(V0, name="uy")

    expr_ux = fem.Expression(flux_solution[0], V0.element.interpolation_points)
    expr_uy = fem.Expression(flux_solution[1], V0.element.interpolation_points)

    ux_fun.interpolate(expr_ux)
    uy_fun.interpolate(expr_uy)

    ux = np.zeros((nx, ny), dtype=np.float64)
    uy = np.zeros((nx, ny), dtype=np.float64)
    ux[ii, jj] = ux_fun.x.array
    uy[ii, jj] = uy_fun.x.array

    pressure_array = np.zeros((nx, ny), dtype=np.float64)
    pressure_array[ii, jj] = pressure_solution.x.array

    mass_balance = compute_boundary_fluxes(flux_solution, darcy_data)

    if return_diagnostics:
        return pressure_solution, ux, uy, pressure_array, flux_solution, mass_balance

    return pressure_solution, ux, uy

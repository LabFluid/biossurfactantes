from pathlib import Path
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem


msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [3.67, 1.0]], [220, 60], mesh.CellType.quadrilateral)
V = fem.functionspace(msh, ("CG", 1))

facets_left = mesh.locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 0.0))
facets_right = mesh.locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 3.67))

dofs_left = fem.locate_dofs_topological(V, msh.topology.dim-1, facets_left)
dofs_right = fem.locate_dofs_topological(V, msh.topology.dim-1, facets_right)
                                             
bc_left = fem.dirichletbc(ScalarType(10.0), dofs_left, V)
bc_right = fem.dirichletbc(ScalarType(0.0), dofs_right, V)
bcs = [bc_left, bc_right]

with open ("spe10_layer_36.pbt") as f:
    data = f.readlines()
permeability = np.array([float(x) for x in data]).reshape(60, 220).T

V_K = fem.functionspace(msh, ("DG", 0))
K = fem.Function(V_K)
K.name = "Permeability"

idx_i = np.floor(permeability.shape[0])
idx_j = np.floor(permeability.shape[1])

dof_coords = V_K.tabulate_dof_coordinates()
idx_i = np.clip(np.floor(dof_coords[:, 0] / (3.67/220)).astype(int), 0, 219)
idx_j = np.clip(np.floor(dof_coords[:, 1] / (1.0/60)).astype(int), 0, 59)
K.x.array[:] = permeability[idx_i, idx_j].flatten()

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.dot(K * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(msh, 0.0) * v * ufl.dx

problem = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="darcy_flow_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "Pressure"
assert isinstance(uh, fem.Function)

W = fem.functionspace(msh, ("DG", 0, (msh.topology.dim,)))
u_vel = fem.Function(W, name="Velocity")

flux_expr = fem.Expression(-K * ufl.grad(uh), W.element.interpolation_points)
u_vel.interpolate(flux_expr)

out_folder = Path("output")
out_folder.mkdir(exist_ok=True)
out_file = out_folder / "porous_media.bp"
with io.VTXWriter(msh.comm, out_file, [uh, K, u_vel]) as vtx:
    vtx.write(0.0)
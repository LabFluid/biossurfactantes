# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Poisson equation
#
# This demo illustrates how to:
#
# - Create a {py:class}`function space <dolfinx.fem.FunctionSpace>`
# - Solve a linear partial differential equation
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_poisson.py>`
# * {download}`Jupyter notebook <./demo_poisson.ipynb>`
# ```
# ## Equation and problem definition
#
# For a domain $\Omega \subset \mathbb{R}^n$ with boundary $\partial
# \Omega = \Gamma_{D} \cup \Gamma_{N}$, the Poisson equation with
# particular boundary conditions reads:
#
# $$
# \begin{align}
#   - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
#   u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
#   \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
# \end{align}
# $$
#
# where $f$ and $g$ are input data and $n$ denotes the outward directed
# boundary normal. The variational problem reads: find $u \in V$ such
# that
#
# $$
# a(u, v) = L(v) \quad \forall \ v \in V,
# $$
#
# where $V$ is a suitable function space and
#
# $$
# \begin{align}
#   a(u, v) &:= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
#   L(v)    &:= \int_{\Omega} f v \, {\rm d} x + \int_{\Gamma_{N}} g v \,
#               {\rm d} s.
# \end{align}
# $$
#
# The expression $a(u, v)$ is the bilinear form and $L(v)$
# is the linear form. It is assumed that all functions in $V$
# satisfy the Dirichlet boundary conditions ($u = 0 \ {\rm on} \
# \Gamma_{D}$).
#
# In this demo we consider:
#
# - $\Omega = [0,2] \times [0,1]$ (a rectangle)
# - $\Gamma_{D} = \{(0, y) \cup (2, y) \subset \partial \Omega\}$
# - $\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}$
# - $g = \sin(5x)$
# - $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$
#
# ## Implementation
#
# The modules that will be used are imported:

# +
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem

# -

# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.

# We create a rectangular {py:class}`Mesh <dolfinx.mesh.Mesh>` using
# {py:func}`create_rectangle <dolfinx.mesh.create_rectangle>`, and
# create a finite element {py:class}`function space
# <dolfinx.fem.FunctionSpace>` $V$ on the mesh.

# +
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (3.67, 1.0)),
    n=(220, 60),
    cell_type=mesh.CellType.quadrilateral,
)
V = fem.functionspace(msh, ("Lagrange", 1))
# -

# The second argument to {py:func}`functionspace
# <dolfinx.fem.functionspace>` is a tuple `(family, degree)`, where
# `family` is the finite element family, and `degree` specifies the
# polynomial degree. In this case `V` is a space of continuous Lagrange
# finite elements of degree 1. For further details of how one can specify
# finite elements as tuples, see {py:class}`ElementMetaData
# <dolfinx.fem.ElementMetaData>`.
#
# To apply the Dirichlet boundary conditions, we find the mesh facets
# (entities of topological co-dimension 1) that lie on the boundary
# $\Gamma_D$ using {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>`. The function is provided
# with a 'marker' function that returns `True` for points `x` on the
# boundary and `False` otherwise.

facets_left = mesh.locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 0.0))
facets_right = mesh.locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 3.67))

dofs_left = fem.locate_dofs_topological(V, msh.topology.dim-1, facets_left)
dofs_right = fem.locate_dofs_topological(V, msh.topology.dim-1, facets_right)

bc_left = fem.dirichletbc(ScalarType(10.0), dofs_left, V)
bc_right = fem.dirichletbc(ScalarType(0.0), dofs_right, V)
bcs = [bc_left, bc_right]

# Next, the variational problem is defined:

with open("spe10_layer_36.pbt") as f:
    data = f.readlines()
permeability = np.array([float(val) for val in data]).reshape((60, 220)).T 


# Create DG0 Function for Permeability (K)
V_K = fem.functionspace(msh, ("DG", 0))
K = fem.Function(V_K)
K.name = "Permeabilidade"

dof_coords = V_K.tabulate_dof_coordinates()
idx_i = (dof_coords[:, 0] / (3.67/220)).astype(int)
idx_j = (dof_coords[:, 1] / (1.0/60)).astype(int)
K.x.array[:] = permeability[idx_i, idx_j].flatten()

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.dot(K * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(msh, 0.0) * v * ufl.dx 

# A {py:class}`LinearProblem <dolfinx.fem.petsc.LinearProblem>` object is
# created that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case an LU solver is used, and we ask that PETSc throws an error
# if the solver does not converge. The {py:func}`solve
# <dolfinx.fem.petsc.LinearProblem.solve>` computes the solution.

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options_prefix="demo_poisson_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)
uh = problem.solve()
uh.name = "Pressao"
assert isinstance(uh, fem.Function)


W = fem.functionspace(msh, ("DG", 0, (msh.topology.dim,)))
u_vel = fem.Function(W, name="Velocidade")

flux_expr = fem.Expression(-K * ufl.grad(uh), W.element.interpolation_points)
u_vel.interpolate(flux_expr)

# The solution can be written to a {py:class}`XDMFFile
# <dolfinx.io.XDMFFile>` file visualization with [ParaView](https://www.paraview.org/)
# or [VisIt](https://visit-dav.github.io/visit-website/):

out_folder = Path(__file__).parent.absolute() / "out_poisson"
out_folder.mkdir(parents=True, exist_ok=True) 

with io.XDMFFile(msh.comm, out_folder / "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)  
    file.write_function(K)     
    file.write_function(u_vel) 

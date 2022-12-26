import itertools as it

import dolfinx
import numpy as np
import pytest
from fenicsx_plotly import plot
from mpi4py import MPI

# from pathlib import Path


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def get_mesh(dim):
    if dim == 2:
        return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    elif dim == 3:
        return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.mark.parametrize("dim, wireframe", it.product([2, 3], [True, False]))
def test_plot_mesh(dim, wireframe):
    mesh = get_mesh(dim)
    plot(mesh, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, degree",
    it.product([2, 3], [True, False], [1, 2, 3]),
)
def test_plot_scalar_cg_function_space(dim, wireframe, degree):
    mesh = get_mesh(dim)
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", degree))
    plot(V, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, scatter, degree",
    it.product([2, 3], [True, False], [True, False], [1, 2]),
)
def test_plot_scalar_cg_function(dim, wireframe, scatter, degree):
    mesh = get_mesh(dim)
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", degree))
    p = dolfinx.fem.Function(V)
    p.interpolate(lambda x: np.sin(x[0]))
    plot(p, scatter=scatter, wireframe=wireframe, show=False)

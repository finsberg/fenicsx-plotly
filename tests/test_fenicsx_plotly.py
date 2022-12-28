import itertools as it

import dolfinx
import numpy as np
import pytest
import ufl
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
def test_plot_scalar_P_function_space(dim, wireframe, degree):
    mesh = get_mesh(dim)
    V = dolfinx.fem.FunctionSpace(mesh, ("P", degree))
    plot(V, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, scatter, degree",
    it.product([2, 3], [True, False], [True, False], [1, 2]),
)
def test_plot_scalar_P_function(dim, wireframe, scatter, degree):
    mesh = get_mesh(dim)
    V = dolfinx.fem.FunctionSpace(mesh, ("P", degree))
    p = dolfinx.fem.Function(V)
    p.interpolate(lambda x: np.sin(x[0]))
    plot(p, scatter=scatter, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, degree",
    it.product([2, 3], [True, False], [1, 2, 3]),
)
def test_plot_vector_P_function_space(dim, wireframe, degree):
    mesh = get_mesh(dim)
    el = ufl.VectorElement("P", mesh.ufl_cell(), degree)
    V = dolfinx.fem.FunctionSpace(mesh, el)
    plot(V, wireframe=wireframe, show=False)


@pytest.mark.parametrize(
    "dim, wireframe, normalize, degree, component",
    it.product(
        [2, 3],
        [True, False],
        [True, False],
        [1, 2],
        [None, "magnitude", "x", "y", "z"],
    ),
)
def test_plot_vector_cg_function(dim, wireframe, normalize, degree, component):
    mesh = get_mesh(dim)
    el = ufl.VectorElement("P", mesh.ufl_cell(), degree)
    V = dolfinx.fem.FunctionSpace(mesh, el)
    u = dolfinx.fem.Function(V)

    x = ufl.SpatialCoordinate(mesh)

    if dim == 2:
        expr = dolfinx.fem.Expression(
            ufl.as_vector((1 + x[0], x[1])),
            V.element.interpolation_points(),
        )
        if component == "z":
            return
    else:
        expr = dolfinx.fem.Expression(
            ufl.as_vector((1 + x[0], x[1], x[2])),
            V.element.interpolation_points(),
        )

    u.interpolate(expr)
    plot(
        u,
        wireframe=wireframe,
        show=False,
        component=component,
        normalize=normalize,
    )


def test_facet_function_3d():
    mesh = get_mesh(3)
    locator = lambda x: np.isclose(x[0], 0)
    entities = dolfinx.mesh.locate_entities(mesh, 2, locator)
    marker = 1
    values = np.full_like(entities, marker)

    facet_tags = dolfinx.mesh.meshtags(mesh, 2, entities, values)
    plot(facet_tags)

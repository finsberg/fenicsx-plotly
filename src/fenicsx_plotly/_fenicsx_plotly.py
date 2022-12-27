import os
from pathlib import Path

import dolfinx
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import ufl
from petsc4py import PETSc

try:
    _SHOW_PLOT = bool(int(os.getenv("FENICS_PLOTLY_SHOW", 1)))
except ValueError:
    _SHOW_PLOT = True

try:
    _RENDERER = os.getenv("FENICS_PLOTLY_RENDERER", "notebook")
except ValueError:
    _RENDERER = "notebook"


def set_renderer(renderer):
    pio.renderers.default = renderer


set_renderer(_RENDERER)


def project(
    expr: ufl.core.expr.Expr,
    V: dolfinx.fem.FunctionSpace,
) -> dolfinx.fem.Function:
    # Ensure we have a mesh and attach to measure
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = dolfinx.fem.form(ufl.inner(Pv, w) * dx)
    L = dolfinx.fem.form(ufl.inner(expr, w) * dx)

    # Assemble linear system
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    u = dolfinx.fem.Function(V)
    solver.solve(b, u.vector)
    return u


def savefig(fig, filename, save_config=None):
    """Save figure to file

    Parameters
    ----------
    fig : `plotly.graph_objects.Figure`
        This figure that you want to save
    filename : Path or str
        Path to the destination where you want to
        save the figure
    save_config : dict, optional
        Additional configurations to be passed
        to `plotly.offline.plot`, by default None
    """

    filename = Path(filename)
    outdir = filename.parent
    assert outdir.exists(), f"Folder {outdir} does not exist"

    config = {
        "toImageButtonOptions": {
            "filename": filename.stem,
            "width": 1500,
            "height": 1200,
        },
    }
    if save_config is not None:
        config.update(save_config)

    path = outdir.joinpath(filename)
    plotly.offline.plot(fig, filename=path.as_posix(), auto_open=False, config=config)


def _get_triangles(mesh):
    faces = dolfinx.mesh.locate_entities(
        mesh,
        2,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    )

    mesh.topology.create_connectivity(2, 0)
    conn = mesh.topology.connectivity(2, 0)
    triangle = np.zeros((3, faces.size), dtype=int)

    for face in faces:
        # FIXME: Should be possible to do this vectorized!
        triangle[:, face] = conn.links(face)

    return triangle


def _surface_plot_mesh(mesh, color, opacity=1.0, **kwargs):
    coord = mesh.geometry.x
    triangle = _get_triangles(mesh)
    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    surface = go.Mesh3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        i=triangle[0, :],
        j=triangle[1, :],
        k=triangle[2, :],
        flatshading=True,
        color=color,
        opacity=opacity,
        lighting=dict(ambient=1),
    )

    return surface


def _get_cells(mesh) -> np.ndarray:
    dm = mesh.geometry.dofmap
    cells = np.zeros((dm.num_nodes, len(dm.links(0))), dtype=np.int32)
    # FIXME: Should be possible to vectorize this
    for node in range(dm.num_nodes):
        cells[node, :] = dm.links(node)
    return cells


def _wireframe_plot_mesh(mesh, **kwargs):
    coord = mesh.geometry.x

    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    cells = _get_cells(mesh)
    tri_points = coord[cells]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])

    # define the trace for triangle sides
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        name="",
        line=dict(color="rgb(70,70,70)", width=2),
        hoverinfo="none",
    )

    return lines


def _plot_dofs(functionspace: dolfinx.fem.FunctionSpace, size: int, **kwargs):
    dofs_coord = functionspace.tabulate_dof_coordinates()
    if len(dofs_coord[0, :]) == 2:
        dofs_coord = np.c_[dofs_coord, np.zeros(len(dofs_coord[:, 0]))]

    points = go.Scatter3d(
        x=dofs_coord[:, 0],
        y=dofs_coord[:, 1],
        z=dofs_coord[:, 2],
        mode="markers",
        name=kwargs.get("name", None),
        marker=dict(size=size),
    )

    return points


def _get_vertex_values(function: dolfinx.fem.Function) -> np.ndarray:

    fs = function.function_space
    mesh = fs.mesh
    shape = function.ufl_shape

    if len(shape) == 0:  # FiniteElement
        el = fs.ufl_element()
        # TODO: Ask Jørgen if there is a better way
        # Where is the `.compute_vertex_values()` method?
        if (el.family(), el.degree()) != ("P", 1):
            # Interpolate into a linear lagrange space
            V = dolfinx.fem.FunctionSpace(mesh, ("P", 1))
            u = dolfinx.fem.Function(V)
            u.interpolate(function)
            res = u.x.array
        else:
            res = function.x.array
    elif len(shape) == 1:  # Vector Element
        res = np.zeros((mesh.geometry.x.shape[0], shape[0]))
        for i in range(shape[0]):
            res[:, i] = _get_vertex_values(function.sub(i).collapse())

    else:  # Tensor Element
        res = np.zeros((mesh.geometry.x.shape[0], shape[0], shape[1]))
        count = 0
        for i in range(shape[0]):
            for j in range(shape[0]):
                res[:, i, j] = _get_vertex_values(function.sub(count + j).collapse())
            count += shape[0]
    return res


def _surface_plot_function(
    function, colorscale, showscale=True, intensitymode="vertex", **kwargs
):
    fs = function.function_space
    mesh = fs.mesh

    val = _get_vertex_values(function=function)

    triangle = _get_triangles(mesh)

    coord = mesh.geometry.x

    hoverinfo = ["val:" + "%.5f" % item for item in val]

    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    surface = go.Mesh3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        i=triangle[0, :],
        j=triangle[1, :],
        k=triangle[2, :],
        flatshading=True,
        intensitymode=intensitymode,
        intensity=val,
        colorscale=colorscale,
        lighting=dict(ambient=1),
        name="",
        hoverinfo="all",
        text=hoverinfo,
        showscale=showscale,
    )

    return surface


def _scatter_plot_function(
    function: dolfinx.fem.Function, colorscale, showscale=True, size=10, **kwargs
):
    dofs_coord = function.function_space.tabulate_dof_coordinates()
    if len(dofs_coord[0, :]) == 2:
        dofs_coord = np.c_[dofs_coord, np.zeros(len(dofs_coord[:, 0]))]

    mesh = function.function_space.mesh
    val = function.x.array
    coord = mesh.geometry.x
    hoverinfo = ["val:" + "%.5f" % item for item in val]

    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    points = go.Scatter3d(
        x=dofs_coord[:, 0],
        y=dofs_coord[:, 1],
        z=dofs_coord[:, 2],
        mode="markers",
        marker=dict(size=size, color=val, colorscale=colorscale),
        hoverinfo="all",
        text=hoverinfo,
    )

    return points


def _cone_plot(function, size=10, showscale=True, normalize=False, **kwargs):

    mesh = function.function_space.mesh
    points = mesh.geometry.x
    vectors = _get_vertex_values(function)

    if len(points[0, :]) == 2:
        points = np.c_[points, np.zeros(len(points[:, 0]))]

    if vectors.shape[1] == 2:
        vectors = np.c_[vectors, np.zeros(len(vectors[:, 0]))]

    if normalize:
        vectors = np.divide(vectors.T, np.linalg.norm(vectors, axis=1)).T

    cones = go.Cone(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=vectors[:, 0],
        v=vectors[:, 1],
        w=vectors[:, 2],
        sizemode="absolute",
        sizeref=size,
        showscale=showscale,
    )

    return cones


def _handle_mesh(obj, **kwargs):
    data = []
    wireframe = bool(kwargs.get("wireframe", False))
    if not wireframe:
        surf = _surface_plot_mesh(obj, **kwargs)
        data.append(surf)

    data.append(_wireframe_plot_mesh(obj))

    return data


def _handle_function_space(obj, **kwargs):
    data = []
    points = _plot_dofs(obj, **kwargs)
    data.append(points)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.mesh, **kwargs)
        data.append(lines)
    return data


def _handle_function(
    obj,
    **kwargs,
):
    data = []
    scatter = kwargs.get("scatter", False)
    norm = kwargs.get("norm", False)
    component = kwargs.get("component", None)
    fs = obj.function_space

    if len(obj.ufl_shape) == 0:
        if scatter:
            surface = _scatter_plot_function(obj, **kwargs)
        else:
            surface = _surface_plot_function(obj, **kwargs)
        data.append(surface)

    elif len(obj.ufl_shape) == 1:
        if norm or component == "magnitude":
            V, _ = obj.function_space.sub(0).collapse()
            magnitude = dolfinx.fem.Function(V)
            magnitude = project(ufl.sqrt(ufl.inner(obj, obj)), V)
        else:
            magnitude = None

        if component is None:
            if norm:
                surface = _surface_plot_function(magnitude, **kwargs)
                data.append(surface)

            cones = _cone_plot(obj, **kwargs)
            data.append(cones)
        else:
            if component == "magnitude":
                surface = _surface_plot_function(magnitude, **kwargs)
                data.append(surface)
            else:

                for i, comp in enumerate(["x", "y", "z"]):

                    if component not in [comp, comp.upper()]:
                        continue
                    if i >= obj.function_space.num_sub_spaces:
                        raise RuntimeError(
                            f"Cannot extract component from subspace {i} for"
                            f" function space with {fs.num_sub_spaces}"
                            " number of subspaces.",
                        )
                    surface = _surface_plot_function(obj.sub(i).collapse(), **kwargs)
                    data.append(surface)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.function_space.mesh)
        data.append(lines)

    return data


def plot(
    obj,
    colorscale="inferno",
    wireframe=True,
    scatter=False,
    size=10,
    norm=False,
    name="f",
    color="gray",
    opacity=1.0,
    show_grid=False,
    size_frame=None,
    background=(242, 242, 242),
    normalize=False,
    component=None,
    showscale=True,
    show=True,
    filename=None,
):
    """Plot FEniCSx object

    Parameters
    ----------
    obj : Mesh, Function. FunctionSpace, MeshFunction, DirichletBC
        FEniCSx object to be plotted
    colorscale : str, optional
        The colorscale, by default "inferno"
    wireframe : bool, optional
        Whether you want to show the mesh in wireframe, by default True
    scatter : bool, optional
        Plot function as scatter plot, by default False
    size : int, optional
        Size of scatter points, by default 10
    norm : bool, optional
        For vectors plot the norm as a surface, by default False
    name : str, optional
        Name to show up in legend, by default "f"
    color : str, optional
        Color to be plotted on the mesh, by default "gray"
    opacity : float, optional
        opacity of surface, by default 1.0
    show_grid : bool, optional
        Show x, y (and z) axis grid, by default False
    size_frame : [type], optional
        Size of plot, by default None
    background : tuple, optional
        Background of plot, by default (242, 242, 242)
    normalize : bool, optional
        For vectors, normalize then to have unit length, by default False
    component : [type], optional
        Plot a component (["Magnitude", "x", "y", "z"]) for vector, by default None
    showscale : bool, optional
        Show colorbar, by default True
    show : bool, optional
        Show figure, by default True
    filename : [type], optional
        Path to file where you want to save the figure, by default None

    Raises
    ------
    TypeError
        If object to be plotted is not recognized.
    """

    if isinstance(obj, dolfinx.mesh.Mesh):
        handle = _handle_mesh

    elif isinstance(obj, dolfinx.fem.Function):
        handle = _handle_function

    # elif isinstance(obj, fe.cpp.mesh.MeshFunctionSizet):
    #     handle = _handle_meshfunction

    elif isinstance(obj, dolfinx.fem.FunctionSpace):
        handle = _handle_function_space

    # elif isinstance(obj, fe.DirichletBC):
    #     handle = _handle_dirichlet_bc

    else:
        raise TypeError(f"Cannot plot object of type {type(obj)}")

    data = handle(
        obj,
        scatter=scatter,
        colorscale=colorscale,
        norm=norm,
        normalize=normalize,
        size=size,
        size_frame=size_frame,
        component=component,
        opacity=opacity,
        show_grid=show_grid,
        color=color,
        wireframe=wireframe,
        showscale=showscale,
        name=name,
    )

    layout = go.Layout(
        scene_xaxis_visible=show_grid,
        scene_yaxis_visible=show_grid,
        scene_zaxis_visible=show_grid,
        paper_bgcolor="rgb" + str(background),
        margin=dict(l=80, r=80, t=50, b=50),
        scene=dict(aspectmode="data"),
    )

    if size_frame is not None:
        layout.update(width=size_frame[0], height=size_frame[1])

    fig = go.FigureWidget(data=data, layout=layout)
    fig.update_layout(hovermode="closest")

    if filename is not None:
        savefig(fig, filename)

    if show and _SHOW_PLOT:
        fig.show()

    return FEniCSPlotFig(fig)


class FEniCSPlotFig:
    def __init__(self, fig):
        self.figure = fig

    def add_plot(self, fig):
        data = list(self.figure.data) + list(fig.figure.data)
        self.figure = go.FigureWidget(data=data, layout=self.figure.layout)

    def show(self):
        if _SHOW_PLOT:
            self.figure.show()

    def save(self, filename):
        savefig(self.figure, filename)

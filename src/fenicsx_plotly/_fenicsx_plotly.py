import os
import typing
from pathlib import Path

import dolfinx
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import ufl
from plotly.basedatatypes import BaseTraceType as _BaseTraceType

try:
    _SHOW_PLOT = bool(int(os.getenv("FENICS_PLOTLY_SHOW", 1)))
except ValueError:
    _SHOW_PLOT = True

try:
    _RENDERER = os.getenv("FENICS_PLOTLY_RENDERER", "notebook")
except ValueError:
    _RENDERER = "notebook"


def set_renderer(renderer: str) -> None:
    pio.renderers.default = renderer


set_renderer(_RENDERER)


def savefig(
    fig: go.FigureWidget,
    filename: str,
    save_config: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
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

    fname = Path(filename)
    outdir = fname.parent
    assert outdir.exists(), f"Folder {outdir} does not exist"

    config = {
        "toImageButtonOptions": {
            "filename": fname.stem,
            "width": 1500,
            "height": 1200,
        },
    }
    if save_config is not None:
        config.update(save_config)

    plotly.offline.plot(fig, filename=fname.as_posix(), auto_open=False, config=config)


def _get_triangles(mesh: dolfinx.mesh.Mesh) -> np.ndarray[int]:
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


def _surface_plot_mesh(
    mesh: dolfinx.mesh.Mesh, color: str = "gray", opacity: float = 1.0, **kwargs
):
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


def _get_cells(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    dm = mesh.geometry.dofmap
    cells = np.zeros((dm.num_nodes, len(dm.links(0))), dtype=np.int32)
    # FIXME: Should be possible to vectorize this
    for node in range(dm.num_nodes):
        cells[node, :] = dm.links(node)
    return cells


def _wireframe_plot_mesh(mesh: dolfinx.mesh.Mesh, **kwargs) -> go.Scatter3d:
    coord = mesh.geometry.x

    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    cells = _get_cells(mesh)

    X = []
    Y = []
    Z = []
    for c in cells:
        X.extend(coord[c, :][:, 0].tolist() + [None])
        Y.extend(coord[c, :][:, 1].tolist() + [None])
        Z.extend(coord[c, :][:, 2].tolist() + [None])

    # define the trace for triangle sides
    lines = go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        mode="lines",
        name="",
        line=dict(color="rgb(70,70,70)", width=2),
        hoverinfo="none",
    )

    return lines


def _plot_dofs(
    functionspace: dolfinx.fem.FunctionSpace, size: int, **kwargs
) -> go.Scatter3d:
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
    function: dolfinx.fem.Function,
    colorscale: str = "inferno",
    showscale: bool = True,
    intensitymode: str = "vertex",
    **kwargs,
) -> go.Mesh3d:
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
) -> go.Scatter3d:
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


def _cone_plot(
    function: dolfinx.fem.Function,
    size: int = 10,
    showscale: bool = True,
    normalize: bool = False,
    **kwargs,
) -> go.Cone:

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


def _handle_mesh(obj: dolfinx.mesh.Mesh, **kwargs) -> list[_BaseTraceType]:
    data = []
    wireframe = bool(kwargs.get("wireframe", False))
    if not wireframe:
        surf = _surface_plot_mesh(obj, **kwargs)
        data.append(surf)

    data.append(_wireframe_plot_mesh(obj))

    return data


def _handle_function_space(
    obj: dolfinx.fem.FunctionSpace, **kwargs
) -> list[_BaseTraceType]:
    data = []
    points = _plot_dofs(obj, **kwargs)
    data.append(points)

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.mesh, **kwargs)
        data.append(lines)
    return data


def _handle_scalar_function(
    obj: dolfinx.fem.Function, scatter: bool = False, **kwargs
) -> _BaseTraceType:
    if scatter:
        surface = _scatter_plot_function(obj, **kwargs)
    else:
        surface = _surface_plot_function(obj, **kwargs)
    return surface


def _handle_vector_function(
    obj: dolfinx.fem.Function,
    component: typing.Optional[str] = None,
    **kwargs,
) -> _BaseTraceType:

    fs = obj.function_space

    if component is None:
        return _cone_plot(obj, **kwargs)

    elif component == "magnitude":
        V, _ = obj.function_space.sub(0).collapse()
        magnitude = dolfinx.fem.Function(V)
        magnitude.interpolate(
            dolfinx.fem.Expression(
                ufl.sqrt(ufl.inner(obj, obj)),
                V.element.interpolation_points(),
            ),
        )
        return _surface_plot_function(magnitude, **kwargs)

    else:
        # Extract x, y or z
        i = {"x": 0, "y": 1, "z": 2}[component.lower()]
        if i >= fs.num_sub_spaces:
            raise RuntimeError(
                f"Cannot extract component from subspace {i} for"
                f" function space with {fs.num_sub_spaces}"
                " number of subspaces.",
            )
        return _surface_plot_function(obj.sub(i).collapse(), **kwargs)


def _handle_function(
    obj: dolfinx.fem.Function,
    **kwargs,
) -> list[_BaseTraceType]:
    data = []

    if len(obj.ufl_shape) == 0:  # Scalar Function
        data.append(_handle_scalar_function(obj, **kwargs))

    elif len(obj.ufl_shape) == 1:  # Vector Function
        data.append(_handle_vector_function(obj, **kwargs))

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(obj.function_space.mesh)
        data.append(lines)

    return data


def _handle_meshtags(
    obj: dolfinx.mesh.MeshTagsMetaClass, colorscale: str = "inferno", **kwargs
) -> list[_BaseTraceType]:

    data = []
    if obj.dim != 2:
        raise NotImplementedError("Plotting of MeshTags is only supported for facets")
    mesh = obj.mesh
    # array = meshfunc.array()
    coord = mesh.geometry.x
    if len(coord[0, :]) == 2:
        coord = np.c_[coord, np.zeros(len(coord[:, 0]))]

    triangle = _get_triangles(mesh)
    array = np.zeros(triangle.shape[1])
    array[obj.indices] = obj.values

    hoverinfo = ["val:" + "%d" % item for item in array]

    data.append(
        go.Mesh3d(
            x=coord[:, 0],
            y=coord[:, 1],
            z=coord[:, 2],
            i=triangle[0, :],
            j=triangle[1, :],
            k=triangle[2, :],
            flatshading=True,
            intensity=array,
            colorscale=colorscale,
            lighting=dict(ambient=1),
            name="",
            hoverinfo="all",
            text=hoverinfo,
            intensitymode="cell",
        ),
    )

    if kwargs.get("wireframe", True):
        lines = _wireframe_plot_mesh(mesh)
        data.append(lines)

    return data


def _plot_dirichlet_bc(
    obj: dolfinx.fem.bcs.DirichletBCMetaClass,
    size: int = 10,
    colorscale: str = "inferno",
    **kwargs,
) -> list[_BaseTraceType]:
    if obj.function_space.element.num_sub_elements > 0:
        raise NotImplementedError(
            "Can plot dirichlet BC for finite elements (not vector elements)",
        )
    dofs = obj.function_space.tabulate_dof_coordinates()
    if len(dofs[0, :]) == 2:
        dofs = np.c_[dofs, np.zeros(len(dofs[:, 0]))]
    indices, _ = obj.dof_indices()

    coords = dofs[indices]
    vals = obj.value.x.array[indices]

    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=vals,
            colorscale=colorscale,
            colorbar=dict(thickness=20),
        ),
    )


def _handle_dirichlet_bc(
    obj: dolfinx.fem.bcs.DirichletBCMetaClass, **kwargs
) -> list[_BaseTraceType]:
    data = []
    points = _plot_dirichlet_bc(obj, **kwargs)
    data.append(points)

    lines = _wireframe_plot_mesh(obj.function_space.mesh, **kwargs)
    data.append(lines)
    return data


class FEniCSPlotFig:
    def __init__(self, fig: go.FigureWidget) -> None:
        self.figure = fig

    def add_plot(self, fig: go.FigureWidget) -> None:
        data = list(self.figure.data) + list(fig.figure.data)
        self.figure = go.FigureWidget(data=data, layout=self.figure.layout)

    def show(self) -> None:
        if _SHOW_PLOT:
            self.figure.show()

    def save(self, filename: str) -> None:
        savefig(self.figure, filename)


def plot(
    obj,
    colorscale: str = "inferno",
    wireframe: bool = True,
    scatter: bool = False,
    size: int = 10,
    name: str = "f",
    color: str = "gray",
    opacity: float = 1.0,
    show_grid: bool = False,
    size_frame: typing.Optional[typing.Tuple[int, int]] = None,
    background: typing.Tuple[int, int, int] = (242, 242, 242),
    normalize: bool = False,
    component: typing.Optional[str] = None,
    showscale: bool = True,
    show: bool = True,
    filename: typing.Optional[str] = None,
) -> FEniCSPlotFig:
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

    elif isinstance(obj, dolfinx.mesh.MeshTagsMetaClass):
        handle = _handle_meshtags

    elif isinstance(obj, dolfinx.fem.FunctionSpace):
        handle = _handle_function_space

    elif isinstance(obj, dolfinx.fem.bcs.DirichletBCMetaClass):
        handle = _handle_dirichlet_bc

    else:
        raise TypeError(f"Cannot plot object of type {type(obj)}")

    data = handle(
        obj,
        scatter=scatter,
        colorscale=colorscale,
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

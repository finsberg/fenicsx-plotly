import os

import dolfinx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# from pathlib import Path
# import plotly

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


def _get_triangles(mesh):
    faces = dolfinx.mesh.locate_entities(
        mesh,
        2,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    )

    mesh.topology.create_connectivity(2, 1)
    conn = mesh.topology.connectivity(2, 1)
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


def _handle_mesh(obj, **kwargs):
    data = []
    wireframe = kwargs.get("wireframe", True)
    if wireframe:
        surf = _surface_plot_mesh(obj, **kwargs)
        data.append(surf)

    data.append(_wireframe_plot_mesh(obj))

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
    """Plot FEnICS object

    Parameters
    ----------
    obj : Mesh, Function. FunctionoSpace, MeshFunction, DirichleyBC
        FEnicS object to be plotted
    colorscale : str, optional
        The colorscale, by default "inferno"
    wireframe : bool, optional
        Whether you want to show the mesh in wirteframe, by default True
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
        Plot a componenent (["Magnitude", "x", "y", "z"]) for vector, by default None
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

    # elif isinstance(obj, fe.Function):
    #     handle = _handle_function

    # elif isinstance(obj, fe.cpp.mesh.MeshFunctionSizet):
    #     handle = _handle_meshfunction

    # elif isinstance(obj, fe.FunctionSpace):
    #     handle = _handle_function_space

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

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(hovermode="closest")

    # if filename is not None:
    #     savefig(fig, filename)
    if show and _SHOW_PLOT:
        fig.show()
    return fig
    # return FEniCSPlotFig(fig)


class FEniCSPlotFig:
    def __init__(self, fig):
        self.figure = fig

    def add_plot(self, fig):
        data = list(self.figure.data) + list(fig.figure.data)
        self.figure = go.FigureWidget(data=data, layout=self.figure.layout)

    def show(self):
        if _SHOW_PLOT:
            self.figure.show()

    # def save(self, filename):
    #     savefig(self.figure, filename)

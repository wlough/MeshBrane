import numpy as np
import mayavi.mlab as mlab

# def test_triangular_mesh():
# """An example of a cone, ie a non-regular mesh defined by its
#     triangles.
# """
n = 8
t = np.linspace(-np.pi, np.pi, n)
z = np.exp(1j * t)
x = z.real.copy()
y = z.imag.copy()
z = np.zeros_like(x)

triangles = [(0, i, i + 1) for i in range(1, n)]
x = np.r_[0, x]
y = np.r_[0, y]
z = np.r_[1, z]
t = np.r_[0, t]

mlab.triangular_mesh(x, y, z, triangles, scalars=t)
mlab.show()
# %%
from python.half_edge_mesh import Brane
import numpy as np

# from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_base_viewer import MeshViewer


# VertTri2HalfEdgeMeshConverter._oblatify_the_spheres(ratio=0.9)

vf_ply = "./data/ply/binary/torus_003072_vf.ply"
he_ply = "./data/ply/binary/torus_003072_he.ply"
he_ply = "./data/half_edge_base/ply/unit_sphere_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_003072_he.ply"

# he_ply = "./data/half_edge_base/ply/dumbbell_he.ply"
he_ply = "./data/half_edge_base/ply/neovius_coarse_he.ply"
b = Brane.from_half_edge_ply(he_ply)
V = b.xyz_coord_V
F = b.V_of_F
target_faces = 1000
boundary_vertex_deletion = True
# %%
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def vtk_decimate(V, F, target_faces=1000, boundary_vertex_deletion=True):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    num_faces = F.shape[0]
    num_vertices = V.shape[0]

    if num_faces <= target_faces:
        print("num_faces <= target_faces")
        V_indices = np.arange(V.shape[0])
        (
            V_simp,
            F_simp,
        ) = (
            V,
            F,
        )
        # return V, F, V_indices

    else:

        target_reduction = 1 - target_faces / num_faces

        # Create vtkPoints
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(V))

        # Create vtkCellArray for faces
        faces = vtk.vtkCellArray()
        for face in F:
            faces.InsertNextCell(3, face)

        # Create vtkPolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(faces)

        # Add original point IDs to the mesh
        original_ids = numpy_to_vtk(
            np.arange(num_vertices), deep=True, array_type=vtk.VTK_ID_TYPE
        )
        original_ids.SetName("V_indices")
        polydata.GetPointData().AddArray(original_ids)

        # Simplify the mesh
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(polydata)
        decimate.SetTargetReduction(target_reduction)
        decimate.PreserveTopologyOn()
        if not boundary_vertex_deletion:
            decimate.BoundaryVertexDeletionOff()
        decimate.Update()

        # Get the simplified mesh
        simplified_polydata = decimate.GetOutput()

        # Extract simplified vertices and faces
        V_simp = vtk.util.numpy_support.vtk_to_numpy(
            simplified_polydata.GetPoints().GetData()
        )
        F_simp = []
        simplified_faces = simplified_polydata.GetPolys()
        simplified_faces.InitTraversal()
        id_list = vtk.vtkIdList()
        while simplified_faces.GetNextCell(id_list):
            F_simp.append([id_list.GetId(j) for j in range(id_list.GetNumberOfIds())])
        F_simp = np.array(F_simp)

        # Extract the indices of the original vertices used in the simplified mesh
        V_indices = vtk.util.numpy_support.vtk_to_numpy(
            simplified_polydata.GetPointData().GetArray("V_indices")
        )
        V_indices = np.unique(V_indices)
        return V_simp, F_simp, V_indices


bs = Brane.from_vf_data(V_simp, F_simp)
mv = MeshViewer(*bs.data_arrays)
mv.simple_plot()
# %%
from dataclasses import dataclass

import numpy

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    vtkMinimalStandardRandomSequence,
    vtkPoints,
    vtkUnsignedCharArray,
)
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkTriangle
from vtkmodules.vtkFiltersCore import vtkCleanPolyData
from vtkmodules.vtkFiltersModeling import (
    vtkButterflySubdivisionFilter,
    vtkLoopSubdivisionFilter,
)
from vtkmodules.vtkInteractionWidgets import vtkTextRepresentation, vtkTextWidget
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextActor,
    vtkTextProperty,
)

points = vtkPoints()
triangles = vtkCellArray()
dir(points)


def main():
    nc = vtkNamedColors()

    # Make a 32 x 32 grid
    size = 32

    rn = vtkMinimalStandardRandomSequence(seed=1)

    # Define z values for the topography (random height)
    topography = numpy.zeros([size, size])
    for i in range(size):
        for j in range(size):
            topography[i][j] = rn.GetRangeValue(0, 5)
            rn.Next()

    # Define points, triangles and colors
    colors = vtkUnsignedCharArray(number_of_components=3)
    points = vtkPoints()
    triangles = vtkCellArray()

    # Build the meshgrid manually
    count = 0
    for i in range(size - 1):
        for j in range(size - 1):
            z1 = topography[i][j]
            z2 = topography[i][j + 1]
            z3 = topography[i + 1][j]

            # Triangle 1
            points.InsertNextPoint(i, j, z1)
            points.InsertNextPoint(i, (j + 1), z2)
            points.InsertNextPoint((i + 1), j, z3)

            triangle = vtkTriangle()
            triangle.point_ids.SetId(0, count)
            triangle.point_ids.SetId(1, count + 1)
            triangle.point_ids.SetId(2, count + 2)

            triangles.InsertNextCell(triangle)

            z1 = topography[i][j + 1]
            z2 = topography[i + 1][j + 1]
            z3 = topography[i + 1][j]

            # Triangle 2
            points.InsertNextPoint(i, (j + 1), z1)
            points.InsertNextPoint((i + 1), (j + 1), z2)
            points.InsertNextPoint((i + 1), j, z3)

            triangle = vtkTriangle()
            triangle.point_ids.SetId(0, count + 3)
            triangle.point_ids.SetId(1, count + 4)
            triangle.point_ids.SetId(2, count + 5)

            count += 6

            triangles.InsertNextCell(triangle)

            # Add some color.
            r = [int(i / float(size) * 255), int(j / float(size) * 255), 0]
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)

    # Create a polydata object.
    # Adding the geometry and topology to the polydata.
    triangle_poly_data = vtkPolyData(points=points, polys=triangles)
    triangle_poly_data.point_data.SetScalars(colors)

    # Clean the polydata so that the edges are shared!
    clean_poly_data = vtkCleanPolyData()
    triangle_poly_data >> clean_poly_data

    # Use a filter to smooth the data (will add triangles and smooth).
    # Use two different filters to show the difference.
    smooth_loop = vtkLoopSubdivisionFilter(number_of_subdivisions=3)

    smooth_butterfly = vtkButterflySubdivisionFilter(number_of_subdivisions=3)

    # Create a mapper and actor for the initial dataset.
    mapper = vtkPolyDataMapper()
    clean_poly_data >> mapper
    actor = vtkActor(mapper=mapper, position=(0, 8, 0))

    # Create a mapper and actor for smoothed dataset (vtkLoopSubdivisionFilter).
    mapper = vtkPolyDataMapper()
    clean_poly_data >> smooth_loop >> mapper
    actor_loop = vtkActor(mapper=mapper, position=(0, 8, 0))

    # Create a mapper and actor for smoothed dataset (vtkButterflySubdivisionFilter).
    mapper = vtkPolyDataMapper()
    clean_poly_data >> smooth_butterfly >> mapper
    actor_butterfly = vtkActor(mapper=mapper, position=(0, 8, 0))

    render_window = vtkRenderWindow(size=(900, 300))
    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.render_window = render_window

    text = {0: "Initial Terrain", 1: "Loop Subdivision", 2: "Butterfly Subdivision"}

    # Define viewport ranges [x_min, y_min, x_max, y_max]
    viewports = {
        0: [0.0, 0.0, 1.0 / 3.0, 1.0],
        1: [1.0 / 3.0, 0.0, 2.0 / 3.0, 1.0],
        2: [2.0 / 3.0, 0.0, 1.0, 1.0],
    }
    camera = None
    # Build the renderers and add them to the render window.
    renderers = list()
    for k in text.keys():
        renderers.append(
            vtkRenderer(background=nc.GetColor3d("Cornsilk"), viewport=viewports[k])
        )

        # Add the actors.
        if k == 0:
            renderers[k].AddActor(actor)
        elif k == 1:
            renderers[k].AddActor(actor_loop)
        elif k == 2:
            renderers[k].AddActor(actor_butterfly)

        if k == 0:
            camera = renderers[k].active_camera
            camera.Elevation(-45)
        else:
            renderers[k].active_camera = camera

        renderers[k].ResetCamera()
        camera.Zoom(1.2)

        render_window.AddRenderer(renderers[k])

    # Create the TextActors.
    text_actors = list()
    text_representations = list()
    text_widgets = list()
    text_property = vtkTextProperty(
        color=nc.GetColor3d("DarkSlateGray"),
        bold=True,
        italic=False,
        shadow=False,
        font_size=12,
        font_family_as_string="Courier",
        justification=TextProperty.Justification.VTK_TEXT_CENTERED,
        vertical_justification=TextProperty.VerticalJustification.VTK_TEXT_CENTERED,
    )

    text_positions = get_text_positions(
        list(text.values()),
        justification=TextProperty.Justification.VTK_TEXT_CENTERED,
        vertical_justification=TextProperty.VerticalJustification.VTK_TEXT_BOTTOM,
    )

    for k, v in text.items():
        text_actors.append(
            vtkTextActor(
                input=v,
                text_scale_mode=vtkTextActor.TEXT_SCALE_MODE_NONE,
                text_property=text_property,
            )
        )

        # Create the text representation. Used for positioning the text actor.
        text_representations.append(
            vtkTextRepresentation(enforce_normalized_viewport_bounds=True)
        )
        text_representations[k].GetPositionCoordinate().value = text_positions[v]["p"]
        text_representations[k].GetPosition2Coordinate().value = text_positions[v]["p2"]

        # Create the TextWidget
        text_widgets.append(
            vtkTextWidget(
                representation=text_representations[k],
                text_actor=text_actors[k],
                default_renderer=renderers[k],
                interactor=render_window_interactor,
                selectable=False,
            )
        )

    render_window.Render()

    for k in text.keys():
        text_widgets[k].On()

    render_window_interactor.Start()


def get_text_positions(
    names, justification=0, vertical_justification=0, width=0.96, height=0.1
):
    """
    Get viewport positioning information for a list of names.

    :param names: The list of names.
    :param justification: Horizontal justification of the text, default is left.
    :param vertical_justification: Vertical justification of the text, default is bottom.
    :param width: Width of the bounding_box of the text in screen coordinates.
    :param height: Height of the bounding_box of the text in screen coordinates.
    :return: A list of positioning information.
    """
    # The gap between the left or right edge of the screen and the text.
    dx = 0.02
    width = abs(width)
    if width > 0.96:
        width = 0.96

    y0 = 0.01
    height = abs(height)
    if height > 0.9:
        height = 0.9
    dy = height
    if vertical_justification == TextProperty.VerticalJustification.VTK_TEXT_TOP:
        y0 = 1.0 - (dy + y0)
        dy = height
    if vertical_justification == TextProperty.VerticalJustification.VTK_TEXT_CENTERED:
        y0 = 0.5 - (dy / 2.0 + y0)
        dy = height

    name_len_min = 0
    name_len_max = 0
    first = True
    for k in names:
        sz = len(k)
        if first:
            name_len_min = name_len_max = sz
            first = False
        else:
            name_len_min = min(name_len_min, sz)
            name_len_max = max(name_len_max, sz)
    text_positions = dict()
    for k in names:
        sz = len(k)
        delta_sz = width * sz / name_len_max
        if delta_sz > width:
            delta_sz = width

        if justification == TextProperty.Justification.VTK_TEXT_CENTERED:
            x0 = 0.5 - delta_sz / 2.0
        elif justification == TextProperty.Justification.VTK_TEXT_RIGHT:
            x0 = 1.0 - dx - delta_sz
        else:
            # Default is left justification.
            x0 = dx

        # For debugging!
        # print(
        #     f'{k:16s}: (x0, y0) = ({x0:3.2f}, {y0:3.2f}), (x1, y1) = ({x0 + delta_sz:3.2f}, {y0 + dy:3.2f})'
        #     f', width={delta_sz:3.2f}, height={dy:3.2f}')
        text_positions[k] = {"p": [x0, y0, 0], "p2": [delta_sz, dy, 0]}

    return text_positions


@dataclass(frozen=True)
class TextProperty:
    @dataclass(frozen=True)
    class Justification:
        VTK_TEXT_LEFT: int = 0
        VTK_TEXT_CENTERED: int = 1
        VTK_TEXT_RIGHT: int = 2

    @dataclass(frozen=True)
    class VerticalJustification:
        VTK_TEXT_BOTTOM: int = 0
        VTK_TEXT_CENTERED: int = 1
        VTK_TEXT_TOP: int = 2


if __name__ == "__main__":
    main()

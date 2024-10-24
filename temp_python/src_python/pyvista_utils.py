import numpy as np
from temp_python.src_python.global_vars import INT_TYPE, FLOAT_TYPE
import pyvista as pv


def decimate_VF(V, F, target_faces=1000, boundary_vertex_deletion=True):
    """
    Decimate a mesh represented by vertices and faces.
    """
    num_faces = F.shape[0]
    num_vertices = V.shape[0]
    #
    if num_faces <= target_faces:
        print("num_faces <= target_faces")
        V_indices = np.arange(V.shape[0])
        return V, F, V_indices
    target_reduction = 1 - target_faces / num_faces

    F_pv = np.zeros((num_faces, 4), dtype=INT_TYPE)
    F_pv[:, 0] = 3
    F_pv[:, 1:] = F
    F_pv = F_pv.ravel()

    # Create a PyVista mesh
    M = pv.PolyData(V, F_pv)

    # Add original point IDs to the mesh
    M.point_data["V_indices"] = np.arange(num_vertices)

    # Simplify the mesh (reduce the number of faces) and preserve original point IDs
    Msimp = M.decimate_pro(
        target_reduction,
        preserve_topology=True,
        boundary_vertex_deletion=boundary_vertex_deletion,
    )
    Msimp = Msimp.smooth(n_iter=20, relaxation_factor=0.01)

    # Extract simplified vertices and faces
    Vsimp = np.array(Msimp.points)
    Fsimp = Msimp.faces.reshape(-1, 4)[:, 1:]

    # Extract the indices of the original vertices used in the simplified mesh
    indices_Vsimp = np.unique(Msimp.point_data["V_indices"])
    return Vsimp, Fsimp, indices_Vsimp


def vtk_decimate_VF(V, F, target_faces=1000, boundary_vertex_deletion=True):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    num_faces = F.shape[0]
    num_vertices = V.shape[0]

    if num_faces <= target_faces:
        print("num_faces <= target_faces")
        V_indices = np.arange(V.shape[0])
        return V, F, V_indices

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


def fix_VF(V, F, target_faces=1000, boundary_vertex_deletion=True):
    """
    Decimate a mesh represented by vertices and faces.
    """
    import numpy as np
    import pyvista as pv

    num_faces = F.shape[0]
    num_vertices = V.shape[0]

    F_pv = np.zeros((num_faces, 4), dtype="int32")
    F_pv[:, 0] = 3
    F_pv[:, 1:] = F
    F_pv = F_pv.ravel()
    # Create a PyVista mesh
    M = pv.PolyData(V, F_pv)
    M = M.clean(point_merging=True)
    M.plot(show_edges=True)
    # Apply Laplacian smoothing
    M = M.smooth(n_iter=20, relaxation_factor=0.01)
    M.plot(show_edges=True)
    # Apply Loop subdivision
    M = M.subdivide(2, subfilter="loop")
    target_faces = num_faces
    target_reduction = 1 - target_faces / M.number_of_cells
    M = M.decimate(
        target_reduction,
    )
    # M = M.decimate_pro(
    #     target_reduction,
    #     preserve_topology=True,
    #     boundary_vertex_deletion=True,
    # )
    M.plot(show_edges=True)

    # Extract simplified vertices and faces
    Vsimp = np.array(M.points)
    Fsimp = M.faces.reshape(-1, 4)[:, 1:]

    # Extract the indices of the original vertices used in the simplified mesh
    indices_Vsimp = np.unique(M.point_data["V_indices"])
    return Vsimp, Fsimp, indices_Vsimp

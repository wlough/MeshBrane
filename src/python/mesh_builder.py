from plyfile import PlyData, PlyElement, PlyListProperty, PlyProperty
import numpy as np

# dtype_dict = {"signed 32-bit integer": {"numpy": "int32", "ply": "int"}, "unsigned 32-bit integer": {"numpy": "uint32", "ply": "uint"}, "double": {"numpy": "uint32", "ply": "int"}}


class StandardTetrahedron:
    """
    name : str
        Name of the cell.
    order : int
        Number of vertices in the cell.
    dimension : int
        Dimension of the cell.
    V : list
        List of vertex coordinates.
    E : list
        List of vertex indices for edges.
    F : list
        List of vertex indices for faces (triangles).


    Vertex A: (0, 0, 0)
    Vertex B: (1, 0, 0)
    Vertex C: (0.5, sqrt(3)/2, 0)
    Vertex D: (0.5, sqrt(3)/6, sqrt(2/3))
    """

    def __init__(self):
        self.name = "tetrahedron"
        self.order = 4
        self.dimension = 3
        self.V = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.86602540378, 0.0],
            [0.5, 0.28867513459, 0.81649658092],
        ]
        self.E = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
        self.F = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]


class PlySchema:
    """
    Schema for a ply file representing a mesh with certain elements and properties. Default schema is for a 3D mesh represented by vertex/face lists.

    Attributes
    ----------
    name : str
        Name of the schema.
    elements : list of dicts
        List of element names.
    """

    def __init__(
        self,
        name="PlySchema",
        format="ascii 1.0",
        elements=None,
    ):
        self.name = name
        self.format = format
        if elements is not None:
            self.elements = elements
        else:
            self.elements = [dict()]

    def generate_header(self):
        header = ["ply"]
        header.append("format ascii 1.0")
        for element in self.elements:
            header.append(f"element {element['name']} {element['number']}")
            for prop in element["properties"]:
                header.append(f"property {prop['dtype']} {prop['name']}")
            for list_prop in element["list_properties"]:
                header.append(f"property list {list_prop['dtype']} {list_prop['name']}")
        header.append("end_header")
        return header


class StandardVertexTriSchema(PlySchema):
    """
    Schema for a ply file representing a mesh with certain elements and properties. Default schema is for a 3D mesh represented by vertex/face lists.

    Attributes
    ----------
    name : str
        Name of the schema.
    elements : list of dicts
        List of dictionaries describing the elements in the schema.
    """

    def __init__(
        self,
        # format="ascii 1.0",
        # Nvertices=0,
        # Nfaces=0,
    ):
        name = "StandardVertexTri"
        format = "ascii 1.0"
        elements = [
            {
                "name": "vertex",
                # "number": Nvertices,
                "properties": [("x", "double"), ("y", "double"), ("z", "double")],
            },
            {
                "name": "face",
                # "number": Nfaces,
                "properties": [("vertex_indices", "int32", (3,))],
            },
        ]
        super().__init__(name, format, elements)


class CombinatorialMap2dSchema_v_0_0(PlySchema):
    """
    Combinatorial map for a 2d surface mesh.

    Nvertices = len(positions)
    Ndarts = len(origin)
    Nnext_cycles = len(next_cycle) #cycles=#faces
    Ntwin_cycles = len(twin_cycle) #cycles=#2*edges-#boundary edges
    """

    def __init__(self):
        name = "CombinatorialMap2d"
        format = "ascii 1.0"
        elements = [
            {
                "name": "vertex",
                # "number": len(positions),
                "properties": [("x", "double"), ("y", "double"), ("z", "double")],
            },
            {
                "name": "dart",
                # "number": len(origin),
                "properties": [("origin_index", "int32")],
            },
            {
                "name": "next_cycle",
                # "number": len(next_cycle),
                "properties": [("dart_indices", "int32", (3,))],
            },
            {
                "name": "twin_cycle",
                # "number": len(twin_cycle),
                "properties": [("dart_indices", "int32", (2,))],
            },
        ]

        super().__init__(name, format, elements)


class PlyLoader:
    """
    Reading/writing ply files

    Attributes
    ----------
    source_ply_path : str
        Path to the source ply file.
    source_ply_data : PlyData
        Data from the source ply file.
    source_ply_header : list
        Header of the source ply file.
    target_ply_path : str
        Path to the target ply file.
    target_ply_data : PlyData
        Data to be written to the target ply file.


    Methods
    -------
    read_ply_header(ply_path)
        Read the header of a ply file.
    add_element_to_target_data(data, name)
        Add an element to target_ply_data.
    add_property_to_target_element(element_name, property_data)
        Add property to an existing element in target_ply_data.
    print_source_header()
        Print the header of the source ply file.
    print_target_header()
        Print the header of target_ply_data
    """

    default_ply_dtypes = {
        "vertex": [("x", "f8"), ("y", "f8"), ("z", "f8")],
        "face": [("vertex_indices", "i4", (3,))],
    }
    mesh_brane_dtypes = {
        "vertex": {
            3: [("x", "f8"), ("y", "f8"), ("z", "f8")],
            2: [("x", "f8"), ("y", "f8")],
        },
        "edge": [("vertex_indices", "i4", (2,))],
        "tri": [("vertex_indices", "i4", (3,))],
        "tet": [("vertex_indices", "i4", (4,))],
        "dart": [("vertex_indices", "i4", (5,))],
    }

    def __init__(self, source_ply_path, target_ply_path=None):
        self._source_ply_path = source_ply_path
        self._source_ply_data = PlyData.read(source_ply_path)
        self._source_ply_header = self.read_ply_header(source_ply_path)
        self._target_ply_path = target_ply_path
        self._target_ply_data = PlyData()

    @property
    def vertex_dtype(self, dim=3):
        if dim == 2:
            return [("x", "f8"), ("y", "f8")]
        elif dim == 3:
            return [("x", "f8"), ("y", "f8"), ("z", "f8")]

    @property
    def unoriented_kcell_dtype(self, k=3):
        return [("vertex_indices", "i4", (k,))]

    @property
    def source_ply_path(self):
        return self._source_ply_path

    @property
    def source_ply_data(self):
        return self._source_ply_data

    @property
    def target_ply_path(self):
        return self._target_ply_path

    @target_ply_path.setter
    def target_ply_path(self, path):
        self._target_ply_path = path

    @property
    def target_ply_data(self):
        return self._target_ply_data

    @target_ply_data.setter
    def target_ply_data(self, ply_data):
        assert isinstance(ply_data, PlyData), "ply_data must be an instance of PlyData"
        self._target_ply_data = ply_data

    def add_element_to_target_data(self, data, name):
        """
        Add an element to target_ply_data. If target_ply_data does not exist, it will be created.

        Parameters
        ----------
        data : structured array
            dtype of data must be specified as a list of tuples (name, type) or (name, type, shape).
        """
        element = PlyElement.describe(data, name)
        self.target_ply_data = PlyData([element])

    def add_property_to_target_element(self, element_name, property_data):
        """
        Add property to an existing element in target_ply_data.

        Parameters
        ----------
        element_name : str
            Name of the element in target_ply_data.
        property_data : structured array
            dtype must be (name, type) or (name, type, shape)
        """
        try:
            element = self.target_ply_data.elements[element_name]
            property_name, property_type = property_data.dtype.fields.keys()
            if len(property_data.dtype.fields[property_name]) == 2:
                property = PlyProperty((property_name, property_type))
            elif len(property_data.dtype.fields[property_name]) == 3:
                property = PlyListProperty(
                    (
                        property_name,
                        property_type,
                        property_data.dtype.fields[property_name][0].shape[0],
                    )
                )

            element.properties.append(property)
        except KeyError:
            print(f"Element {element_name} not found in target_ply_data.")

    @classmethod
    def read_ply_header(cls, ply_path):
        print(f"Reading {ply_path}")
        print(len(f"Reading {ply_path}") * "=")
        with open(ply_path, "r") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
                print(line.strip())
                if line.strip() == "end_header":
                    break
        return lines

    def print_source_header(self):
        for line in self._source_ply_header:
            print(line)

    def print_target_header(self):
        print("ply")
        if self.target_ply_data.byte_order == "=":
            print("format ascii 1.0")
        elif self.target_ply_data.byte_order == "<":
            print("format binary_little_endian 1.0")
        else:
            print("format binary_big_endian 1.0")

        for element in self.target_ply_data.elements:
            print(f"element {element.name} {len(element.data)}")
            for property in element.properties:
                if isinstance(property, PlyListProperty):
                    print(f"property list {property.data_type} {property.name}")
                else:
                    print(f"property {property.data_type} {property.name}")
        print("end_header")

    def write_target_ply(self, path=None, use_ascii=False):
        if path is None:
            path = self.target_ply_path
        if use_ascii:
            self.target_ply_data.write(path, text=True)
        else:
            self.target_ply_data.write(path, byte_order="=")

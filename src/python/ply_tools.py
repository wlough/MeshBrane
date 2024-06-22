from plyfile import PlyData, PlyElement  # , PlyListProperty, PlyProperty
import numpy as np


class PlySchema:
    """
    Schema for a ply file representing a mesh with certain elements and properties.
    Attributes
    ----------
    name : str
        Name of the schema.
    format : str
        Format of the ply file.
    identifier : str
        Identifier for the ply file.
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties

    Methods
    -------
    generate_header()
        Generate the header for the ply file.
    print_header()
        Print the header for the ply file.
    _schema_from_ply_file(file_path, schema_name=None)
        Constructs a new PlySchema object by reading a ply file header.
    ply_data_to_samples(self, ply_data)
        construct a lists of data from a PlyData object using the schema
    samples_to_ply_data(self, *samples, use_binary=False)
        Construct a PlyData object using the schema
    uchar=uint8
    double=float64
    """

    def __init__(
        self,
        name="PlySchema",
        identifier="ply",
        format="ascii 1.0",
        comments=None,
        elements=None,
    ):
        self.name = name
        self.format = format
        self.identifier = identifier
        if elements is not None:
            self.elements = elements
        else:
            # self.elements = [dict()]
            self.elements = [
                {
                    "name": "element_with_properties",
                    "count": "number_of_element_samples",
                    "properties": [("prop0", "dtype_prop0"), ("prop1", "dtype_prop1")],
                },
                {
                    "name": "element_with_list_property",
                    "count": "number_of_element_samples",
                    "properties": [("list_prop", "dtype_of_stuff_in_list_prop", (3,))],
                },
            ]

        self.header = self.generate_header()

    def generate_header(self):
        header = [self.identifier]
        header.append(f"format {self.format}")
        for element in self.elements:
            header.append(f"element {element['name']} {element['count']}")
            for prop in element["properties"]:
                if len(prop) == 2:
                    header.append(f"property {prop[1]} {prop[0]}")
                elif len(prop) == 3:
                    header.append(f"property list uint8 {prop[1]} {prop[0]}")
        header.append("end_header")
        return header

    def print_header(self):
        for line in self.header:
            print(line)

    @classmethod
    def _schema_from_ply_file(cls, file_path, schema_name=None):
        """Constructs a PlySchema object by reading a ply file header. need to manually set size for list properties"""
        if schema_name is None:
            schema_name = file_path
        elements = []
        comments = []
        identifier = None
        format = None
        with open(file_path, "r") as f:
            lines = f.readlines()
            identifier = lines[0].strip()
            i = 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("format"):
                    _, format = line.split(maxsplit=1)
                elif line.startswith("comment"):
                    comments.append(line[8:].strip())
                elif line.startswith("element"):
                    elements.append(dict())
                    _, element_name, element_count = line.split()
                    elements[-1]["name"] = element_name
                    elements[-1]["count"] = element_count
                    elements[-1]["properties"] = []
                elif line.startswith("property"):
                    # prop_line = lines[i].strip()
                    prop_line_split = lines[i].strip().split()
                    if len(prop_line_split) == 3:
                        _, prop_dtype, prop_name = prop_line_split
                        elements[-1]["properties"].append((prop_name, prop_dtype))
                    if len(prop_line_split) == 5:
                        _, _, list_index_dtype, prop_dtype, prop_name = prop_line_split
                        elements[-1]["properties"].append((prop_name, prop_dtype, (3,)))
                elif line.startswith("end_header"):
                    break
                i += 1
        return cls(
            name=schema_name,
            identifier=identifier,
            format=format,
            comments=comments,
            elements=elements,
        )

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema"""
        # ply_data = PlyData.read(file_path)
        V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
            )
        ]
        F = [
            vertex_indices.tolist()
            for vertex_indices in plydata["face"]["vertex_indices"]
        ]
        samples = (
            V,
            F,
        )
        return samples

    def samples_to_ply_data(self, *samples, use_binary=False):
        """Constructs a PlyData object using the schema"""
        (
            V,
            F,
        ) = samples
        V_data = np.array(
            [tuple(v) for v in V],
            dtype=[("x", "double"), ("y", "double"), ("z", "double")],
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", "int32", (3,))])
        F_data["vertex_indices"] = F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData([vertex_element, face_element], text=not use_binary)


class VertexTriListSchema(PlySchema):
    """
    Schema for a ply file representing a mesh by a list of vertex positions and a list of triangles.

    Attributes
    ----------
    name : str
        Name of the schema.
    format : str
        Format of the ply file (default="ascii 1.0").
    identifier : str
        Identifier for the ply file (default="ply").
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties
    """

    def __init__(
        self,
        # format="ascii 1.0",
        # Nvertices=0,
        # Nfaces=0,
    ):
        comments = ["Vertex-triangle mesh with vertex positions and face indices"]
        elements = [
            {
                "name": "vertex",
                "count": 7,
                "properties": [("x", "double"), ("y", "double"), ("z", "double")],
            },
            {
                "name": "face",
                "count": 6,
                "properties": [("vertex_indices", "int32", (3,))],
            },
        ]
        super().__init__(
            name="StandardVertexTri",
            identifier="ply",
            format="ascii 1.0",
            comments=comments,
            elements=elements,
        )

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema"""
        # ply_data = PlyData.read(file_path)
        V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
            )
        ]
        F = [
            vertex_indices.tolist()
            for vertex_indices in plydata["face"]["vertex_indices"]
        ]
        samples = (
            V,
            F,
        )
        return samples

    def samples_to_ply_data(self, *samples, use_binary=False):
        """Constructs a PlyData object using the schema"""
        (
            V,
            F,
        ) = samples
        V_data = np.array(
            [tuple(v) for v in V],
            dtype=[("x", "double"), ("y", "double"), ("z", "double")],
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", "int32", (3,))])
        F_data["vertex_indices"] = F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData([vertex_element, face_element], text=not use_binary)


class HalfEdgeSchema(PlySchema):
    """
    Schema for a ply file representing HalfEdgeMesh

    Attributes
    ----------
    name : str
        Name of the schema.
    format : str
        Format of the ply file (default="ascii 1.0").
    identifier : str
        Identifier for the ply file (default="ply").
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties
    """

    def __init__(self):
        identify = "he_ply"
        comments = ["Schema for HalfEdgeMesh ply."]
        elements = [
            {
                "name": "vertex",
                "count": "Nvertex",
                "properties": [
                    ("x", "double"),
                    ("y", "double"),
                    ("z", "double"),
                    ("h", "int32"),
                ],
            },
            {
                "name": "half_edge",
                "count": "Nhalf_edge",
                "properties": [
                    ("v", "int32"),
                    ("f", "int32"),
                    ("n", "int32"),
                    ("t", "int32"),
                ],
            },
            {
                "name": "face",
                "count": "Nface",
                "properties": [("h", "int32")],
            },
        ]

        # super().__init__(name, format, elements)
        super().__init__(
            name="CombinatorialMap2d",
            identifier="ply",
            format="ascii 1.0",
            comments=comments,
            elements=elements,
        )

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema
        _xyz_coord_V : list of numpy.array
            _xyz_coord_V[i] = xyz coordinates of vertex i

        _h_out_V : list of int
            _h_out_V[i] = some outgoing half-edge incident on vertex i
        _v_origin_H : list of int
            _v_origin_H[j] = vertex at the origin of half-edge j
        _h_next_H : list of int
            _h_next_H[j] next half-edge after half-edge j in the face cycle
        _h_twin_H : list of int
            _h_twin_H[j] = half-edge antiparalel to half-edge j
        _f_left_H : list of int
            _f_left_H[j] = face to the left of half-edge j
            _f_left_H[j] = -1 if half-edge j is on a boundary of the mesh

        _h_bound_F : list of int
            _h_bound_F[k] = some half-edge on the boudary of face k
        """
        # ply_data = PlyData.read(file_path)
        _xyz_coord_V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
            )
        ]
        _h_out_V = plydata["vertex"]["h"].tolist()
        _v_origin_H = plydata["half_edge"]["v"].tolist()
        _h_next_H = plydata["half_edge"]["n"].tolist()
        _h_twin_H = plydata["half_edge"]["t"].tolist()
        _f_left_H = plydata["half_edge"]["f"].tolist()
        _h_bound_F = plydata["face"]["h"].tolist()
        samples = (
            _xyz_coord_V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        )
        return samples

    def samples_to_ply_data(self, *samples, use_binary=False):
        """Constructs a PlyData object using the schema"""
        (
            _xyz_coord_V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = samples
        V_data = np.array(
            [(xyz[0], xyz[1], xyz[2], h) for xyz, h in zip(_xyz_coord_V, _h_out_V)],
            dtype=[("x", "double"), ("y", "double"), ("z", "double"), ("h", "int32")],
        )
        H_data = np.array(
            [
                (v, n, t, f)
                for v, n, t, f in zip(_v_origin_H, _h_next_H, _h_twin_H, _f_left_H)
            ],
            dtype=[("v", "int32"), ("n", "int32"), ("t", "int32"), ("f", "int32")],
        )
        F_data = np.array(_h_bound_F, dtype=[("h", "int32")])

        vertex_element = PlyElement.describe(V_data, "vertex")
        half_edge_element = PlyElement.describe(H_data, "half_edge")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData(
            [vertex_element, half_edge_element, face_element], text=not use_binary
        )


class HalfEdgeSchema_no_bdry_twin(PlySchema):
    """
    Schema for a ply file representing HalfEdgeMesh

    Attributes
    ----------
    name : str
        Name of the schema.
    format : str
        Format of the ply file (default="ascii 1.0").
    identifier : str
        Identifier for the ply file (default="ply").
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties
    """

    def __init__(self):
        identify = "he_ply"
        comments = ["Schema for HalfEdgeMesh ply."]
        elements = [
            {
                "name": "vertex",
                "count": "Nvertex",
                "properties": [
                    ("x", "double"),
                    ("y", "double"),
                    ("z", "double"),
                    ("h", "int32"),
                ],
            },
            {
                "name": "half_edge",
                "count": "Nhalf_edge",
                "properties": [
                    ("v", "int32"),
                    ("f", "int32"),
                    ("n", "int32"),
                    ("t", "int32"),
                ],
            },
            {
                "name": "face",
                "count": "Nface",
                "properties": [("h", "int32")],
            },
        ]

        # super().__init__(name, format, elements)
        super().__init__(
            name="CombinatorialMap2d",
            identifier="ply",
            format="ascii 1.0",
            comments=comments,
            elements=elements,
        )

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema
        _xyz_coord_V : list of numpy.array
            _xyz_coord_V[i] = xyz coordinates of vertex i

        _h_out_V : list of int
            _h_out_V[i] = some outgoing half-edge incident on vertex i
        _v_origin_H : list of int
            _v_origin_H[j] = vertex at the origin of half-edge j
        _h_next_H : list of int
            _h_next_H[j] next half-edge after half-edge j in the face cycle
        _h_twin_H : list of int
            _h_twin_H[j] = half-edge antiparalel to half-edge j
            _h_twin_H[j] = -1 if half-edge j is on a boundary of the mesh
        _f_left_H : list of int
            _f_left_H[j] = face to the left of half-edge j

        _h_bound_F : list of int
            _h_bound_F[k] = some half-edge on the boudary of face k
        """
        # ply_data = PlyData.read(file_path)
        _xyz_coord_V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
            )
        ]
        _h_out_V = plydata["vertex"]["h"].tolist()
        _v_origin_H = plydata["half_edge"]["v"].tolist()
        _h_next_H = plydata["half_edge"]["n"].tolist()
        _h_twin_H = plydata["half_edge"]["t"].tolist()
        _f_left_H = plydata["half_edge"]["f"].tolist()
        _h_bound_F = plydata["face"]["h"].tolist()
        samples = (
            _xyz_coord_V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        )
        return samples

    def samples_to_ply_data(self, *samples, use_binary=False):
        """Constructs a PlyData object using the schema"""
        (
            _xyz_coord_V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = samples
        V_data = np.array(
            [(xyz[0], xyz[1], xyz[2], h) for xyz, h in zip(_xyz_coord_V, _h_out_V)],
            dtype=[("x", "double"), ("y", "double"), ("z", "double"), ("h", "int32")],
        )
        H_data = np.array(
            [
                (v, n, t, f)
                for v, n, t, f in zip(_v_origin_H, _h_next_H, _h_twin_H, _f_left_H)
            ],
            dtype=[("v", "int32"), ("n", "int32"), ("t", "int32"), ("f", "int32")],
        )
        F_data = np.array(_h_bound_F, dtype=[("h", "int32")])

        vertex_element = PlyElement.describe(V_data, "vertex")
        half_edge_element = PlyElement.describe(H_data, "half_edge")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData(
            [vertex_element, half_edge_element, face_element], text=not use_binary
        )


class PlyConverter:
    """
    Reading/writing ply files

    Attributes
    ----------
    source_ply_path : str
        Path to the source ply file.
    source_ply_schema : str
        PlySchema for the source ply file.
    source_ply_data : PlyData
        Data from the source ply file.
    source_samples : list
        List of lists of data from the source ply file.
    target_ply_schema : str
        PlySchema for the target ply file.
    target_ply_data : PlyData
        Data from the target ply file.
    target_samples : list
        List of lists of data from the target ply file.

    Methods
    -------
    from_source_ply(cls, source_ply_schema, target_ply_schema, source_ply_path)
        Construct a PlyConverter object from a source ply file.
    write_target_ply(target_path=None, use_ascii=False)
        Write the target ply file.
    read_ply_header(ply_path)
        Read the header of a ply file.
    print_target_ply_data_header
        Print the header of the target_ply_data
    print_source_ply_data_header
        Print the header of the source_ply_data
    source_samples_to_target_samples(*source_samples)
        Override to define how to convert source samples to target samples.
    """

    # source_ply_schema = PlySchema()
    # target_ply_schema

    def __init__(
        self,
        source_ply_schema,
        target_ply_schema,
        source_ply_data=None,
        source_ply_path=None,
    ):
        self._source_ply_schema = source_ply_schema
        self._target_ply_schema = target_ply_schema
        self._source_ply_data = source_ply_data
        if source_ply_data is not None:
            self._source_samples = self.source_ply_schema.ply_data_to_samples(
                self.source_ply_data
            )
            self._target_samples = self.source_samples_to_target_samples(
                *self._source_samples
            )
            self._target_ply_data = self.target_ply_schema.samples_to_ply_data(
                *self._target_samples
            )
        else:
            self._source_samples = None
            self._target_samples = None
            self._target_ply_data = None
        self._source_ply_path = source_ply_path

    @property
    def source_ply_schema(self):
        return self._source_ply_schema

    @property
    def target_ply_schema(self):
        return self._target_ply_schema

    @property
    def source_ply_data(self):
        return self._source_ply_data

    @property
    def source_samples(self):
        return self._source_samples

    @property
    def target_samples(self):
        return self._target_samples

    @property
    def target_ply_data(self):
        return self._target_ply_data

    @property
    def source_ply_path(self):
        return self._source_ply_path

    @classmethod
    def from_source_ply(cls, source_ply_schema, target_ply_schema, source_ply_path):
        source_ply_data = PlyData.read(source_ply_path)
        return cls(source_ply_schema, target_ply_schema, source_ply_data)

    @classmethod
    def from_source_samples(cls, source_ply_schema, target_ply_schema, *source_samples):
        source_ply_data = source_ply_schema.samples_to_ply_data(*source_samples)
        return cls(source_ply_schema, target_ply_schema, source_ply_data)

    @classmethod
    def from_target_samples(cls, source_ply_schema, target_ply_schema, *target_samples):
        pc = cls(
            source_ply_schema,
            target_ply_schema,
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_samples = target_samples
        pc._target_ply_data = target_ply_schema.samples_to_ply_data(*target_samples)
        return pc

    @classmethod
    def from_target_ply(cls, source_ply_schema, target_ply_schema, target_ply_path):
        pc = cls(
            source_ply_schema,
            target_ply_schema,
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_ply_data = PlyData.read(target_ply_path)
        pc._target_samples = pc.target_ply_schema.ply_data_to_samples(
            pc.target_ply_data
        )
        return pc

    def write_target_ply(self, target_path=None, use_ascii=False):
        self.target_ply_data.text = use_ascii
        self.target_ply_data.write(target_path)

    def write_source_ply(self, target_path=None, use_ascii=False):
        self.source_ply_data.text = use_ascii
        self.source_ply_data.write(target_path)

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

    def print_target_ply_data_header(self):
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

    def print_source_ply_data_header(self):
        print("ply")
        if self.source_ply_data.byte_order == "=":
            print("format ascii 1.0")
        elif self.source_ply_data.byte_order == "<":
            print("format binary_little_endian 1.0")
        else:
            print("format binary_big_endian 1.0")

        for element in self.source_ply_data.elements:
            print(f"element {element.name} {len(element.data)}")
            for property in element.properties:
                if isinstance(property, PlyListProperty):
                    print(f"property list {property.data_type} {property.name}")
                else:
                    print(f"property {property.data_type} {property.name}")
        print("end_header")

    def source_samples_to_target_samples(self, *source_samples):
        target_samples = source_samples
        return target_samples


class VertTri2HalfEdgeConverter_no_bdry_twin(PlyConverter):
    """
    Builds a half-edge mesh data from a vertex-triangle mesh.

    Boundary convention: All half-edges have a face to the left. Half-edges adjacent to boundary are assigned a -1 twin.

    Attributes
    ----------
    source_ply_path : str
        Path to the source ply file.
    source_ply_schema : str
        PlySchema for the source ply file.
    source_ply_data : PlyData
        Data from the source ply file.
    source_samples : list
        List of lists of data from the source ply file.
    target_ply_schema : str
        PlySchema for the target ply file.
    target_ply_data : PlyData
        Data from the target ply file.
    target_samples : list
        List of lists of data from the target ply file.

    Methods
    -------
    from_source_ply(cls, source_ply_schema, target_ply_schema, source_ply_path)
        Construct a PlyConverter object from a source ply file.
    write_target_ply(target_path=None, use_ascii=False)
        Write the target ply file.
    read_ply_header(ply_path)
        Read the header of a ply file.
    print_target_ply_data_header
        Print the header of the target_ply_data
    print_source_ply_data_header
        Print the header of the source_ply_data
    source_samples_to_target_samples(*source_samples)
        Override to define how to convert source samples to target samples.
    """

    def __init__(
        self,
        source_ply_data=None,
        source_ply_path=None,
    ):

        source_ply_schema = VertexTriListSchema()
        target_ply_schema = HalfEdgeSchema_no_bdry_twin()
        super().__init__(
            source_ply_schema, target_ply_schema, source_ply_data, source_ply_path
        )

    @classmethod
    def from_source_ply(cls, source_ply_path):
        source_ply_data = PlyData.read(source_ply_path)
        return cls(source_ply_data)

    @classmethod
    def from_source_samples(cls, *source_samples):
        source_ply_schema = VertexTriListSchema()
        source_ply_data = source_ply_schema.samples_to_ply_data(*source_samples)
        return cls(source_ply_data)

    @classmethod
    def from_target_samples(cls, *target_samples):
        target_ply_schema = HalfEdgeSchema()
        pc = cls(
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_samples = target_samples
        pc._target_ply_data = target_ply_schema.samples_to_ply_data(*target_samples)
        return pc

    @classmethod
    def from_target_ply(cls, target_ply_path):
        target_ply_schema = HalfEdgeSchema()
        pc = cls(
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_ply_data = PlyData.read(target_ply_path)
        pc._target_samples = pc.target_ply_schema.ply_data_to_samples(
            pc.target_ply_data
        )
        return pc

    def get_index_of_twin(self, E, e):
        Nedges = len(E)
        v0 = E[e][0]
        v1 = E[e][1]
        for e_twin in range(Nedges):
            if E[e_twin][0] == v1 and E[e_twin][1] == v0:
                return e_twin

        return -1

    def source_samples_to_target_samples(self, *source_samples):
        (V, F) = source_samples
        Nfaces = len(F)
        Nvertices = len(V)
        Nedges = 3 * Nfaces

        E = Nedges * [[0, 0]]

        V_edge = Nvertices * [-1]
        E_vertex = Nedges * [0]
        E_next = Nedges * [0]
        E_twin = Nedges * [-2]
        E_face = Nedges * [0]
        F_edge = Nfaces * [0]

        for f in range(Nfaces):
            F_edge[f] = 3 * f
            for i in range(3):
                e = 3 * f + i
                e_next = 3 * f + (i + 1) % 3
                v0 = F[f][i]
                v1 = F[f][(i + 1) % 3]
                E[e] = [v0, v1]
                E_vertex[e] = v0
                E_face[e] = f
                E_next[e] = e_next
                if V_edge[v0] == -1:
                    V_edge[v0] = e

        for e in range(Nedges):
            if E_twin[e] == -2:
                e_twin = self.get_index_of_twin(E, e)
                E_twin[e] = e_twin
                if e_twin != -1:
                    E_twin[e_twin] = e

        target_samples = (V, V_edge, E_vertex, E_next, E_twin, E_face, F_edge)
        return target_samples


##################################################
class VertTri2HalfEdgeConverter(PlyConverter):
    """
    Builds a half-edge mesh data from a vertex-triangle mesh.

    Boundary convention: All half-edges have a twin. Boundary half-edges are oriented cw are assigned a -1 face.

    Attributes
    ----------
    source_ply_path : str
        Path to the source ply file.
    source_ply_schema : str
        PlySchema for the source ply file.
    source_ply_data : PlyData
        Data from the source ply file.
    source_samples : list
        List of lists of data from the source ply file.
    target_ply_schema : str
        PlySchema for the target ply file.
    target_ply_data : PlyData
        Data from the target ply file.
    target_samples : list
        List of lists of data from the target ply file.

    Methods
    -------
    from_source_ply(cls, source_ply_schema, target_ply_schema, source_ply_path)
        Construct a PlyConverter object from a source ply file.
    write_target_ply(target_path=None, use_ascii=False)
        Write the target ply file.
    read_ply_header(ply_path)
        Read the header of a ply file.
    print_target_ply_data_header
        Print the header of the target_ply_data
    print_source_ply_data_header
        Print the header of the source_ply_data
    source_samples_to_target_samples(*source_samples)
        Override to define how to convert source samples to target samples.
    """

    def __init__(
        self,
        source_ply_data=None,
        source_ply_path=None,
    ):

        source_ply_schema = VertexTriListSchema()
        target_ply_schema = HalfEdgeSchema()
        super().__init__(
            source_ply_schema, target_ply_schema, source_ply_data, source_ply_path
        )

    @classmethod
    def from_source_ply(cls, source_ply_path):
        source_ply_data = PlyData.read(source_ply_path)
        return cls(source_ply_data)

    @classmethod
    def from_source_samples(cls, *source_samples):
        source_ply_schema = VertexTriListSchema()
        source_ply_data = source_ply_schema.samples_to_ply_data(*source_samples)
        return cls(source_ply_data)

    @classmethod
    def from_target_samples(cls, *target_samples):
        target_ply_schema = HalfEdgeSchema()
        pc = cls(
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_samples = target_samples
        pc._target_ply_data = target_ply_schema.samples_to_ply_data(*target_samples)
        return pc

    @classmethod
    def from_target_ply(cls, target_ply_path):
        target_ply_schema = HalfEdgeSchema()
        pc = cls(
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_ply_data = PlyData.read(target_ply_path)
        pc._target_samples = pc.target_ply_schema.ply_data_to_samples(
            pc.target_ply_data
        )
        return pc

    def get_index_of_twin(self, H, h):
        """
        Find the half-edge twin to h in the list of half-edges H.

        Parameters
        ----------
        H : list
            List of half-edges [[v0, v1], ...]
        h : int
            Index of half-edge in H

        Returns
        -------
        h_twin : int
            Index of H[h_twin]=[v1,v0] in H, where H[h]=[v0,v1]. Returns -1 if twin not found.
        """
        Nhedges = len(H)
        v0 = H[h][0]
        v1 = H[h][1]
        for h_twin in range(Nhedges):
            if H[h_twin][0] == v1 and H[h_twin][1] == v0:
                return h_twin

        return -1

    def source_samples_to_target_samples(self, *source_samples):
        (V, F) = source_samples
        Nfaces = len(F)
        Nvertices = len(V)

        H = []
        h_out_V = Nvertices * [-1]
        v_origin_H = []
        h_next_H = []
        f_left_H = []
        h_bound_F = Nfaces * [0]

        # h = 0
        for f in range(Nfaces):
            h_bound_F[f] = 3 * f
            for i in range(3):
                h = 3 * f + i
                h_next = 3 * f + (i + 1) % 3
                v0 = F[f][i]
                v1 = F[f][(i + 1) % 3]
                H.append([v0, v1])
                v_origin_H.append(v0)
                f_left_H.append(f)
                h_next_H.append(h_next)
                if h_out_V[v0] == -1:
                    h_out_V[v0] = h
        need_twins = set([_ for _ in range(len(H))])
        need_next = set()
        h_twin_H = len(H) * [-2]  # -2 means not set
        while need_twins:
            h = need_twins.pop()
            if h_twin_H[h] == -2:  # if twin not set
                h_twin = self.get_index_of_twin(H, h)  # returns -1 if twin not found
                if h_twin == -1:  # if twin not found
                    h_twin = len(H)
                    v0, v1 = H[h]
                    H.append([v1, v0])
                    v_origin_H.append(v1)
                    need_next.add(h_twin)
                    h_twin_H[h] = h_twin
                    h_twin_H.append(h)
                    f_left_H.append(-1)
                else:
                    h_twin_H[h], h_twin_H[h_twin] = h_twin, h
                    need_twins.remove(h_twin)

        h_next_H.extend([-1] * len(need_next))
        while need_next:
            h = need_next.pop()
            h_next = h_twin_H[h]
            # rotate ccw around origin of twin until we find nex h on boundary
            while f_left_H[h_next] != -1:
                h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
            h_next_H[h] = h_next

        # find and enumerate boundaries -1,-2,...
        H_need2visit = set([h for h in range(len(H)) if f_left_H[h] == -1])
        bdry_count = 0
        while H_need2visit:
            bdry_count += 1
            h_start = H_need2visit.pop()
            f_left_H[h_start] = -bdry_count
            h = h_next_H[h_start]
            while h != h_start:
                H_need2visit.remove(h)
                f_left_H[h] = -bdry_count
                h = h_next_H[h]

        target_samples = (
            V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
        return target_samples

    def target_samples_to_source_samples(self, *target_samples):
        (xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H, h_bound_F) = (
            target_samples
        )
        # Nfaces = len(h_bound_F)

        F = []
        for h_start in h_bound_F:
            F.append([])
            h = h_start
            while True:
                F[-1].append(v_origin_H[h])
                h = h_next_H[h]
                if h == h_start:
                    break
        source_samples = (xyz_coord_V, F)
        return source_samples


##################################################
##################################################


class HalfEdgeMeshBuilder:
    """Constructs data to initialize a half-edge mesh from a vertex-face list

    parameters
    ----------
    V: list
        numpy arrays containing xyz coordinates of each vertex
    V_edge: list
        half-edge indices of a half-edge incident on each vertex
    E_vertex: list
        vertex indices for the origin of each half-edge
    E_face: list
        face indices of face to the left of each half-edge
    E_next: list
        half-edge indices for next half-edge
    E_twin: list
        half-edge indices for the twin of each half-edge
    F_edge: list
        half-edge indices of a half-edge on the boudary of each face
    """

    def __init__(self, V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge):
        self.V = [np.copy(xyz) for xyz in V]
        self.V_edge = V_edge.copy()
        self.E_vertex = E_vertex.copy()
        self.E_face = E_face.copy()
        self.E_next = E_next.copy()
        self.E_twin = E_twin.copy()
        self.F_edge = F_edge.copy()

    @classmethod
    def get_index_of_twin(self, E, e):
        Nedges = len(E)
        v0 = E[e][0]
        v1 = E[e][1]
        for e_twin in range(Nedges):
            if E[e_twin][0] == v1 and E[e_twin][1] == v0:
                return e_twin

        return -1

    @classmethod
    def from_vert_tri_lists(cls, V, F):
        Nfaces = len(F)
        Nvertices = len(V)
        Nedges = 3 * Nfaces

        E = Nedges * [[0, 0]]

        V_edge = Nvertices * [-1]
        E_vertex = Nedges * [0]
        E_face = Nedges * [0]
        E_next = Nedges * [0]
        E_twin = Nedges * [-2]
        F_edge = Nfaces * [0]

        for f in range(Nfaces):
            F_edge[f] = 3 * f
            for i in range(3):
                e = 3 * f + i
                e_next = 3 * f + (i + 1) % 3
                v0 = F[f][i]
                v1 = F[f][(i + 1) % 3]
                E[e] = [v0, v1]
                E_vertex[e] = v0
                E_face[e] = f
                E_next[e] = e_next
                if V_edge[v0] == -1:
                    V_edge[v0] = e

        for e in range(Nedges):
            if E_twin[e] == -2:
                e_twin = cls.get_index_of_twin(E, e)
                E_twin[e] = e_twin
                if e_twin != -1:
                    E_twin[e_twin] = e

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    @classmethod
    def from_vertex_tri_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
            )
        ]
        F = [verts.tolist() for verts in plydata["face"]["vertex_indices"]]
        return cls.from_vert_tri_lists(V, F)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["he_vertex"]["x"],
                plydata["he_vertex"]["y"],
                plydata["he_vertex"]["z"],
            )
        ]
        V_edge = plydata["he_vertex"]["e"].tolist()
        F_edge = plydata["he_face"]["e"].tolist()
        E_vertex = plydata["he_edge"]["v"].tolist()
        E_face = plydata["he_edge"]["f"].tolist()
        E_next = plydata["he_edge"]["n"].tolist()
        E_twin = plydata["he_edge"]["t"].tolist()

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    def get_faces(self):
        return [
            [
                self.E_vertex[e],
                self.E_vertex[self.E_next[e]],
                self.E_vertex[self.E_next[self.E_next[e]]],
            ]
            for e in self.F_edge
        ]

    def to_half_edge_ply(self, ply_path, use_binary=False):
        V_data = np.array(
            [
                (vertex[0], vertex[1], vertex[2], e)
                for vertex, e in zip(self.V, self.V_edge)
            ],
            dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("e", "uint32")],
        )
        F_data = np.array(self.F_edge, dtype=[("e", "uint32")])
        E_data = np.array(
            [
                (v, f, n, t)
                for v, f, n, t in zip(
                    self.E_vertex, self.E_face, self.E_next, self.E_twin
                )
            ],
            dtype=[("v", "uint32"), ("f", "uint32"), ("n", "uint32"), ("t", "i4")],
        )
        V_element = PlyElement.describe(V_data, "he_vertex")
        E_element = PlyElement.describe(E_data, "he_edge")
        F_element = PlyElement.describe(F_data, "he_face")
        PlyData([V_element, E_element, F_element], text=not use_binary).write(ply_path)

    def to_vertex_face_ply(self, ply_path, use_binary=False):
        F = self.get_faces()
        V_data = np.array(
            [tuple(v) for v in self.V], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")]
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", "i4", (3,))])
        F_data["vertex_indices"] = F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        PlyData([vertex_element, face_element], text=not use_binary).write(ply_path)

    @classmethod
    def combo_from_vert_tri_lists(cls, V, F):
        Nfaces = len(F)
        Nvertices = len(V)
        Nedges = 3 * Nfaces

        E = Nedges * [[0, 0]]

        V_edge = Nvertices * [-1]
        E_vertex = Nedges * [0]
        E_face = Nedges * [0]
        E_next = Nedges * [0]
        E_twin = Nedges * [-2]
        F_edge = Nfaces * [0]

        for f in range(Nfaces):
            F_edge[f] = 3 * f
            for i in range(3):
                e = 3 * f + i
                e_next = 3 * f + (i + 1) % 3
                v0 = F[f][i]
                v1 = F[f][(i + 1) % 3]
                E[e] = [v0, v1]
                E_vertex[e] = v0
                E_face[e] = f
                E_next[e] = e_next
                if V_edge[v0] == -1:
                    V_edge[v0] = e

        for e in range(Nedges):
            if E_twin[e] == -2:
                e_twin = cls.get_index_of_twin(E, e)
                E_twin[e] = e_twin
                if e_twin != -1:
                    E_twin[e_twin] = e

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)


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


class CombinatorialMap2dSchema(PlySchema):
    """
    Schema for a ply file representing a 2d surface mesh by combinatorial maps.

    Attributes
    ----------
    name : str
        Name of the schema.
    format : str
        Format of the ply file (default="ascii 1.0").
    identifier : str
        Identifier for the ply file (default="ply").
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties



    Nvertices = len(positions)
    Ndarts = len(origin)
    Nnext_cycles = len(next_cycle) #cycles=#faces
    Ntwin_cycles = len(twin_cycle) #cycles=#2*edges-#boundary edges
    """

    def __init__(self):
        comments = [
            "Schema for a ply file representing a 2d surface mesh by combinatorial maps."
        ]
        elements = [
            {
                "name": "vertex",
                "count": "Nvertex",
                "properties": [("x", "double"), ("y", "double"), ("z", "double")],
            },
            {
                "name": "dart",
                "count": "Ndart",
                "properties": [("origin_index", "int32")],
            },
            {
                "name": "next_cycle",
                "count": "Nnext_cycle",
                "properties": [("dart_indices", "int32", (3,))],
            },
            {
                "name": "twin_cycle",
                "count": "twin_cycle",
                "properties": [("dart_indices", "int32", (2,))],
            },
        ]

        # super().__init__(name, format, elements)
        super().__init__(
            name="CombinatorialMap2d",
            identifier="ply",
            format="ascii 1.0",
            comments=comments,
            elements=elements,
        )


class DataElement:
    """ """

    def __init__(self, name, samples, sample_dtypes):
        self.name = name
        self.samples = samples
        self.sample_dtype = sample_dtype
        self.count = len(samples)
        self.structured_array = np.array(samples, dtype=sample_dtype)


class DataSchema:
    """
    Schema for a custom binary data file representing a mesh with certain elements and properties.
    Attributes
    ----------
    name : str
        Name of the schema.
    identifier : str
        Identifier for the ply file.
    format : str
        Format of the ply file.
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties

    Methods
    -------
    generate_header()
        Generate the header for the ply file.
    print_header()
        Print the header for the ply file.
    _schema_from_ply_file(file_path, schema_name=None)
        Constructs a new PlySchema object by reading a ply file header.
    build_data_from_ply(file_path)
        Overridable method to construct a PlyData object/data lists from a ply file using the schema.

    uchar=uint8
    double=float64

    pybit_flags = {
    'int8': 0b0001,
    'uint8': 0b0010,
    'int16': 0b0100,
    'uint16': 0b1000,
    'int32': 0b00010000,
    'uint32': 0b00100000,
    'float32': 0b01000000,
    'float64': 0b10000000,
    }
    py2cpp_dtype = {
    'int8': 'char',
    'uint8': 'unsigned char',
    'int16': 'short',
    'uint16': 'unsigned short',
    'int32': 'int',
    'uint32': 'unsigned int',
    'float32': 'float',
    'float64': 'double',
    }

    """

    def __init__(
        self,
        name="DataSchema",
        identifier="custom",
        format="binary_little_endian 1.0",
        comments=["this is the first line of a comment", "this is another line"],
        elements=None,
    ):
        self.name = name
        self.identifier = identifier
        self.format = format
        self.comments = comments
        if elements is not None:
            self.elements = elements
        else:
            # self.elements = [dict()]
            self.elements = [
                {
                    "name": "elem0",
                    "count": 3,
                    "properties": [
                        ("scalar_prop0", "double"),
                        ("scalar_prop1", "int32"),
                    ],
                    "samples": [[0.0, 2.6, 1.5], [1, 3, 0]],
                },
                {
                    "name": "elem1",
                    "count": 4,
                    "properties": [("list_prop", "int32", (3,))],
                    "a": 1,
                },
            ]

        self.header = self.generate_header()

    def generate_header(self):
        header = [self.identifier]
        header.append(f"format {self.format}")
        for element in self.elements:
            header.append(f"element {element['name']} {element['count']}")
            for prop in element["properties"]:
                if len(prop) == 2:
                    header.append(f"property {prop[1]} {prop[0]}")
                elif len(prop) == 3:
                    header.append(f"property list uint8 {prop[1]} {prop[0]}")
        header.append("end_header")
        return header

    def print_header(self):
        for line in self.header:
            print(line)

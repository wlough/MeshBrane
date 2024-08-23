from plyfile import PlyData, PlyElement, PlyListProperty


################################
@njit(parallel=True)
def jit_vf_samples_to_he_samples(V, F):
    # (V, F) = source_samples
    Nfaces = len(F)
    Nvertices = len(V)

    H = []
    h_out_V = Nvertices * [-1]
    v_origin_H = []
    h_next_H = []
    f_left_H = []
    h_bound_F = np.zeros(Nfaces, dtype=np.int32)

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
            h_twin = jit_get_halfedge_index_of_twin(
                H, h
            )  # returns -1 if twin not found
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
    H_need2visit = set([h for h in range(len(H)) if f_left_H[h] < 0])
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


################################


class MeshSchema:
    """
    Schema for a ply file representing a mesh with certain elements and properties.
    Attributes
    ----------
    name : str
        Name of the schema.
    ply_format : str
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
        name="MeshSchema",
        identifier="ply",
        ply_format="binary_little_endian 1.0",
        comments=None,
        elements=None,
    ):
        self.name = name
        self.ply_format = ply_format
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
        header.append(f"format {self.ply_format}")
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
        ply_format = None
        with open(file_path, "r") as f:
            lines = f.readlines()
            identifier = lines[0].strip()
            i = 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("format"):
                    _, ply_format = line.split(maxsplit=1)
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
            ply_format=ply_format,
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


class VertexTriMeshSchema(MeshSchema):
    """
    Schema for a ply file representing a mesh by a list of vertex positions and a list of triangles.

    Attributes
    ----------
    name : str
        Name of the schema.
    ply_format : str
        Format of the ply file (default="binary_little_endian 1.0").
    identifier : str
        Identifier for the ply file (default="ply").
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties
    float_type : str
        Data type for float properties (default="double").
    int_type : str
        Data type for int properties (default="int32").
    """

    def __init__(self, float_type="double", int_type="int32"):
        comments = [
            "Mesh represented by a list vertex positions and a list of vertex indices that make up each face"
        ]
        elements = [
            {
                "name": "vertex",
                "count": 7,
                "properties": [("x", float_type), ("y", float_type), ("z", float_type)],
            },
            {
                "name": "face",
                "count": 6,
                "properties": [("vertex_indices", int_type, (3,))],
            },
        ]
        super().__init__(
            name="VertexTriMeshSchema",
            identifier="ply",
            ply_format="binary_little_endian 1.0",
            comments=comments,
            elements=elements,
        )
        self.float_type = float_type
        self.int_type = int_type

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema"""
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=self.float_type,
        ).T
        vvv_of_F = np.array(plydata["face"]["vertex_indices"], dtype=self.int_type)
        samples = (
            xyz_coord_V,
            vvv_of_F,
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
            dtype=[
                ("x", self.float_type),
                ("y", self.float_type),
                ("z", self.float_type),
            ],
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", self.int_type, (3,))])
        F_data["vertex_indices"] = F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData([vertex_element, face_element], text=not use_binary)


class HalfEdgeMeshSchema(MeshSchema):
    """
    Schema for a ply file representing HalfEdgeMesh

    Attributes
    ----------
    name : str
        Name of the schema.
    format : str
        Format of the ply file (default="binary_little_endian 1.0").
    identifier : str
        Identifier for the ply file (default="ply").
    elements : list of dicts
        List of dicts containing info about ply elemtents and their properties
    """

    def __init__(self, float_type="double", int_type="int32"):
        identifier = "he_ply"
        comments = ["Schema for HalfEdgeMesh ply with boundary."]
        elements = [
            {
                "name": "vertex",
                "count": "Nvertex",
                "properties": [
                    ("x", float_type),
                    ("y", float_type),
                    ("z", float_type),
                    ("h", int_type),
                ],
            },
            {
                "name": "half_edge",
                "count": "Nhalf_edge",
                "properties": [
                    ("v", int_type),
                    ("f", int_type),
                    ("n", int_type),
                    ("t", int_type),
                ],
            },
            {
                "name": "face",
                "count": "Nface",
                "properties": [
                    ("h", int_type),
                ],
            },
            {
                "name": "boundary",
                "count": "Nboundary",
                "properties": [
                    ("h", int_type),
                ],
            },
        ]

        # super().__init__(name, format, elements)
        super().__init__(
            name="HalfEdgeMeshSchema",
            identifier="ply",
            format="binary_little_endian 1.0",
            comments=comments,
            elements=elements,
        )

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i

        h_out_V : list of int
            h_out_V[i] = some outgoing half-edge incident on vertex i
        v_origin_H : list of int
            v_origin_H[j] = vertex at the origin of half-edge j
        h_next_H : list of int
            h_next_H[j] next half-edge after half-edge j in the face cycle
        h_twin_H : list of int
            h_twin_H[j] = half-edge antiparalel to half-edge j
        f_left_H : list of int
            f_left_H[j] = face to the left of half-edge j
            f_left_H[j] = -(b+1) if half-edge j is contained in boundary b and the complement of the mesh is left of j
        h_bound_F : list of int
            h_bound_F[k] = some half-edge on the boudary of face k
        h_comp_B : list of int
            h_comp_B[b] = some half-edge in the boundary b which is right of the complement of the mesh
        """
        # ply_data = PlyData.read(file_path)
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=self.float_type,
        ).T
        h_out_V = np.array(plydata["vertex"]["h"], dtype=self.int_type)
        v_origin_H = np.array(plydata["half_edge"]["v"], dtype=self.int_type)
        h_next_H = np.array(plydata["half_edge"]["n"], dtype=self.int_type)
        h_twin_H = np.array(plydata["half_edge"]["t"], dtype=self.int_type)
        f_left_H = np.array(plydata["half_edge"]["f"], dtype=self.int_type)
        h_bound_F = np.array(plydata["face"]["h"], dtype=self.int_type)
        samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
        try:
            h_comp_B = np.array(plydata["boundary"]["h"], dtype=self.int_type)
            samples += (h_comp_B,)
        except KeyError:
            pass
        return samples

    def samples_to_ply_data(self, *samples, use_binary=False):
        """Constructs a PlyData object using the schema"""

        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        ) = samples[:7]
        V_data = np.array(
            [(xyz[0], xyz[1], xyz[2], h) for xyz, h in zip(xyz_coord_V, h_out_V)],
            dtype=[
                ("x", self.float_type),
                ("y", self.float_type),
                ("z", self.float_type),
                ("h", self.int_type),
            ],
        )
        H_data = np.array(
            [
                (v, n, t, f)
                for v, n, t, f in zip(v_origin_H, h_next_H, h_twin_H, f_left_H)
            ],
            dtype=[
                ("v", self.int_type),
                ("n", self.int_type),
                ("t", self.int_type),
                ("f", self.int_type),
            ],
        )
        F_data = np.array(h_bound_F, dtype=[("h", self.int_type)])

        vertex_element = PlyElement.describe(V_data, "vertex")
        half_edge_element = PlyElement.describe(H_data, "half_edge")
        face_element = PlyElement.describe(F_data, "face")
        if len(samples) == 8:
            h_comp_B = samples[7]
            B_data = np.array(h_comp_B, dtype=[("h", self.int_type)])
            boundary_element = PlyElement.describe(B_data, "boundary")
            return PlyData(
                [vertex_element, half_edge_element, face_element, boundary_element],
                text=not use_binary,
            )
        return PlyData(
            [vertex_element, half_edge_element, face_element], text=not use_binary
        )


class MeshConverter:
    """
    Reading/writing ply files

    Attributes
    ----------
    source_ply_path : str
        Path to the source ply file.
    source_ply_schema : str
        MeshSchema for the source ply file.
    source_ply_data : PlyData
        Data from the source ply file.
    source_samples : list
        List of lists of data from the source ply file.
    target_ply_schema : str
        MeshSchema for the target ply file.
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


class VertTri2HalfEdgeMeshConverter(MeshConverter):
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

        source_ply_schema = VertexTriMeshSchema()
        target_ply_schema = HalfEdgeMeshSchema()
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
    def jit_from_source_samples(cls, *source_samples):
        source_ply_schema = VertexTriListSchema()
        source_ply_data = source_ply_schema.samples_to_ply_data(*source_samples)
        c = cls()
        c._source_ply_data = source_ply_data
        c._source_samples = c.source_ply_schema.ply_data_to_samples(c.source_ply_data)
        c._target_samples = c.jit_source_samples_to_target_samples(*c._source_samples)
        c._target_ply_data = c.target_ply_schema.samples_to_ply_data(*c._target_samples)
        return c

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

    def jit_index_of_twin(self, H, h):
        return jit_get_halfedge_index_of_twin(H, h)

    def jit_source_samples_to_target_samples(self, *source_samples):
        _V, _F = source_samples
        V = np.array(_V)  # [xyz.tolist() for xyz in _V]
        F = np.array(_F)
        # print(f"{V=}")
        # print(f"{F=}")
        return jit_vf_samples_to_he_samples(V, F)

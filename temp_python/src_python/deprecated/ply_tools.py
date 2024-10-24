from plyfile import PlyData, PlyElement, PlyListProperty  # , #PlyProperty
import numpy as np
from temp_python.src_python.combinatorics import inverse
import sympy as sp
from temp_python.src_python.jit_utils import (
    jit_get_halfedge_index_of_twin,
    jit_vf_samples_to_he_samples,
    jit_refine_icososphere,
    fib_sphere,
)


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


##################################################################
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


##################################################
class SphereFactory:
    """
    subdivides icosahedron (20 triangles; 12 vertices) to create meshes of unit sphere

    after k refinements we have

    |V|=10*4^k+2
    |E|=30*4^k
    |F|=20*4^k
    """

    def __init__(self, name="unit_sphere", jit=False):
        # self._NUM_VERTICES_ = [
        #     12,
        #     42,
        #     162,
        #     642,
        #     2562,
        #     10242,
        #     40962,
        #     163842,
        #     "...",
        #     "...",
        #     "...",
        #     "...",
        #     "...",
        # ]

        self._name = name
        self.jit = jit
        r = 1.0
        self.r = r
        phi = (1.0 + np.sqrt(5.0)) * 0.5  # golden ratio
        _a = 1.0
        _b = 1.0 / phi
        a = r * _a / np.sqrt(_a**2 + _b**2)
        b = r * _b / np.sqrt(_a**2 + _b**2)
        V0 = [
            np.array([0.0, b, -a]),
            np.array([b, a, 0.0]),
            np.array([-b, a, 0.0]),
            np.array([0.0, b, a]),
            np.array([0.0, -b, a]),
            np.array([-a, 0.0, b]),
            np.array([0.0, -b, -a]),
            np.array([a, 0.0, -b]),
            np.array([a, 0.0, b]),
            np.array([-a, 0.0, -b]),
            np.array([b, -a, 0.0]),
            np.array([-b, -a, 0.0]),
        ]
        F0 = [
            [2, 1, 0],
            [1, 2, 3],
            [5, 4, 3],
            [4, 8, 3],
            [7, 6, 0],
            [6, 9, 0],
            [11, 10, 4],
            [10, 11, 6],
            [9, 5, 2],
            [5, 9, 11],
            [8, 7, 1],
            [7, 8, 10],
            [2, 5, 3],
            [8, 1, 3],
            [9, 2, 0],
            [1, 7, 0],
            [11, 9, 6],
            [7, 10, 6],
            [5, 11, 4],
            [10, 8, 4],
        ]

        self._num_vertices = [len(V0)]
        self.F = [F0]
        self.V = V0
        # self.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*self.VF())]
        self.v2h = [self.v2h_from_source_samples()]

    def v2h_from_source_samples(self):
        if self.jit:
            return VertTri2HalfEdgeConverter.jit_from_source_samples(*self.VF())
        else:
            return VertTri2HalfEdgeConverter.from_source_samples(*self.VF())

    def VF(self, level=-1):
        return self.V[: self.num_vertices(level)], self.F[level]

    def num_vertices(self, level=-1):
        return self._num_vertices[level]

    def next_num_vertices(self):
        level = len(self._num_vertices)
        # return self._NUM_VERTICES_[level]
        return 10 * 4**level + 2

    @property
    def name(self):
        return self._name

    def py_refine(self, convert_to_half_edge=True):
        print(f"Refining {self.name}...")
        print(f"num_vertices: {self.num_vertices()}-->{self.next_num_vertices()}")
        F = []
        v_midpt_vv = dict()
        for tri in self.F[-1]:
            v0, v1, v2 = tri
            v01 = v_midpt_vv.get((v0, v1))
            v12 = v_midpt_vv.get((v1, v2))
            v20 = v_midpt_vv.get((v2, v0))
            if v01 is None:
                v01 = len(self.V)
                xyz01 = (self.V[v0] + self.V[v1]) / 2
                xyz01 *= self.r / np.linalg.norm(xyz01)
                self.V.append(xyz01)
                v_midpt_vv[(v0, v1)] = v01
                v_midpt_vv[(v1, v0)] = v01
            if v12 is None:
                v12 = len(self.V)
                xyz12 = (self.V[v1] + self.V[v2]) / 2
                xyz12 *= self.r / np.linalg.norm(xyz12)
                self.V.append(xyz12)
                v_midpt_vv[(v1, v2)] = v12
                v_midpt_vv[(v2, v1)] = v12
            if v20 is None:
                v20 = len(self.V)
                xyz20 = (self.V[v2] + self.V[v0]) / 2
                xyz20 *= self.r / np.linalg.norm(xyz20)
                self.V.append(xyz20)
                v_midpt_vv[(v2, v0)] = v20
                v_midpt_vv[(v0, v2)] = v20
            F.append([v0, v01, v20])
            F.append([v01, v1, v12])
            F.append([v20, v12, v2])
            F.append([v01, v12, v20])
        self.F.append(F)
        self._num_vertices.append(len(self.V))
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            # self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))
            self.v2h.append(self.v2h_from_source_samples())

    def jit_refine(self, convert_to_half_edge=True):
        print(f"Refining {self.name}...")
        print(f"num_vertices: {self.num_vertices()}-->{self.next_num_vertices()}")
        # selfV, F= jit_refine_icososphere(*self.VF(), self.r)
        _V, _F = self.VF()
        V, F = np.array(_V), np.array(_F, dtype=np.int32)
        self.V, _F = jit_refine_icososphere(V, F, self.r)
        self.F.append(_F)

        self._num_vertices.append(len(self.V))
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            # self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))
            self.v2h.append(self.v2h_from_source_samples())

    def refine(self, convert_to_half_edge=True):
        if self.jit:
            self.jit_refine(convert_to_half_edge=convert_to_half_edge)
        else:
            self.py_refine(convert_to_half_edge=convert_to_half_edge)

    def _write_plys(self, level=-1):
        ply_file = f"{self.name}_{self.num_vertices(level):05d}.ply"
        self.v2h[level].write_target_ply(
            f"./data/ply/binary/{ply_file}", use_ascii=False
        )
        self.v2h[level].write_source_ply(f"./data/ply/ascii/{ply_file}", use_ascii=True)

    def write_plys(self, level=-1):
        if isinstance(level, int):
            vf_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_vf.ply"
            )
            he_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_he.ply"
            )
            print(f"Writing vertex-face ply to {vf_path}")
            self.v2h[level].write_source_ply(vf_path, use_ascii=False)
            print(f"Writing half-edge ply to {he_path}")
            self.v2h[level].write_target_ply(he_path, use_ascii=False)

        elif level == "all":
            for level in range(len(self.F)):
                self.write_plys(level=level)
            print(f"Done writing {self.name} plys.")

    @classmethod
    def build_test_plys(cls, num_refine=5, jit=False, name="unit_sphere"):
        b = cls(jit=jit, name=name)
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")
        return b

    @classmethod
    def from_unit_sphere_VF(cls, V, F):
        b = cls()
        b.V = list(V)
        b.F = [list(F)]
        b._num_vertices = [len(V)]
        # b.v2h = [VertTri2HalfEdgeConverter.from_source_samples(V, F[-1])]
        return b

    @classmethod
    def build_noisy_test_plys(cls, num_refine=5, noise_scale=0.01):
        b = cls()
        b._name = "noisy_unit_sphere"
        b.V = [v + b.r * np.random.normal(0, noise_scale, 3) for v in b.V]
        b.V = [b.r * v / np.linalg.norm(v) for v in b.V]
        b.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*b.VF())]
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")

    @classmethod
    def build_fibonacci_test_plys(cls, num_refine=5, noise_scale=0.01):
        b = cls()
        b._name = "fibonacci_sphere"
        b.V = [v + b.r * np.random.normal(0, noise_scale, 3) for v in b.V]
        b.V = [b.r * v / np.linalg.norm(v) for v in b.V]
        b.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*b.VF())]
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")


def VF_torus(p):
    p = 3
    N_big = 3 * 2**p
    N_small = 2**p
    rad_small = 1 / 3
    rad_big = 1

    V = []
    F = []
    for b in range(N_big):
        phi = 2 * np.pi * b / N_big
        bp1 = (b + 1) % N_big
        for s in range(N_small):
            sp1 = (s + 1) % N_small
            phi_small = 2 * np.pi * s / N_small
            x = np.cos(phi) * (rad_big + np.cos(phi_small) * rad_small)
            y = np.sin(phi) * (rad_big + np.cos(phi_small) * rad_small)
            z = np.sin(phi_small) * rad_small
            V.append(np.array([x, y, z]))
            b_s = b * N_small + s
            b_sp1 = b * N_small + sp1
            bp1_s = bp1 * N_small + s
            bp1_sp1 = bp1 * N_small + sp1
            F.append([b_s, bp1_sp1, bp1_s])
            F.append([b_s, b_sp1, bp1_sp1])
    return V, F


class DoughnutFactory0:
    """
    Makes and refines meshes for the tori,
    (sqrt(x**2 + y**2) - rad_big)**2 + z**2 - rad_small**2 = 0.


    Refinements are computed by doubling the number of



    Attributes
    ----------
    name : str
        Name of the mesh (name="torus")
    rad_big : float
        Big radius of the torus. 1 by default.
    rad_small : float
        Small radius of the torus. 1/3 by default.
    p0 : int
        Resolution parameter for lowest level of refinement.
    Nphi : list of int
        Number of azimuthal angle samples in each mesh. Controls resolution along the big circumference. 3 * 2**p by default.
    Npsi : list of int
        Number of samples along the small circumference. 2**p by default.
    ---
    ---
    xyz_coord_V : list of numpy.array
        xyz coordinates of the vertices in the finest mesh.
    F : list of list
        Face list for each mesh level.
    implicit_fun_str : str
        Implicit function for the torus.
    """

    def __init__(self, p0=3, rad_big=1, rad_small=1 / 3):
        self._name = "TESTtorus"
        self.p0 = 3
        pow = self.p0
        N_small = 2**pow
        N_big = 3 * N_small
        rad_big = 1
        rad_small = rad_big / 3
        self.rad_big = rad_big
        self.rad_small = rad_small
        self.implicit_fun_str = (
            f"sqrt(x**2 + y**2) - {self.rad_big})**2 + z**2 - {self.rad_small}**2"
        )

        xyz_coord_V = []
        F = []
        v_BS = []
        for b in range(N_big):
            phi = 2 * np.pi * b / N_big
            bp1 = (b + 1) % N_big
            for s in range(N_small):
                sp1 = (s + 1) % N_small
                phi_small = 2 * np.pi * s / N_small
                x = np.cos(phi) * (rad_big + np.cos(phi_small) * rad_small)
                y = np.sin(phi) * (rad_big + np.cos(phi_small) * rad_small)
                z = np.sin(phi_small) * rad_small
                xyz_coord_V.append(np.array([x, y, z]))
                b_s = b * N_small + s
                b_sp1 = b * N_small + sp1
                bp1_s = bp1 * N_small + s
                bp1_sp1 = bp1 * N_small + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_b_s = b_s
                v_BS.append(v_b_s)

        self.xyz_coord_V = xyz_coord_V
        self.F = [F]
        # self.pow = [pow]
        self.Nphi = [N_big]
        self.Npsi = [N_small]
        self.v_BS = [v_BS]
        self.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*self.VF())]

    def phi_of_xyz(self, x, y, z):
        return np.arctan2(y, x)

    def psi_of_xyz(self, x, y, z):
        return np.arctan2(z, np.sqrt(x**2 + y**2))

    def x_of_phi_psi(self, phi, psi):
        return np.cos(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    def y_of_phi_psi(self, phi, psi):
        return np.sin(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    def z_of_phi_psi(self, phi, psi):
        return np.sin(psi) * self.rad_small

    @classmethod
    def build_test_plys(cls, num_refine=5):
        b = cls()
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")

    @property
    def name(self):
        return self._name

    @property
    def pow(self):
        return [_ for _ in range(3, 3 + len(self.F))]

    def Vindices(self, level=-1):
        return self.v_BS[level]

    def num_vertices(self, level=-1):
        return len(self.Vindices(level))

    def num_faces(self, level=-1):
        return len(self.Vindices(level))

    def VF(self, level=-1):
        F = self.F[level]
        V = [self.xyz_coord_V[v] for v in self.Vindices(level)]
        return V, F

    def refine(self, convert_to_half_edge=True):
        r_b = self.rad_big
        r_s = self.rad_small
        Npsi_coarse = self.Npsi[-1]
        Nphi_coarse = self.Nphi[-1]
        pow_coarse = self.pow[-1]
        Npsi = 2 * Npsi_coarse
        Nphi = 2 * Nphi_coarse
        pow = pow_coarse + 1
        print(f"Refining {self.name}...")
        print(f"num_vertices: {Nphi_coarse*Npsi_coarse}-->{Nphi*Npsi}")
        self.Npsi.append(Npsi)
        self.Nphi.append(Nphi)
        self.pow.append(pow)
        F = []
        v_BS = []
        v_BS_coarse = self.v_BS[-1]

        for b_coarse in range(Nphi_coarse):
            ###################################################
            # add every other vertex to each ring in old mesh
            b = 2 * b_coarse
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s_coarse in range(Npsi_coarse):
                # every other vertex is the same as the coarse mesh
                s = 2 * s_coarse
                b_s_coarse = b_coarse * Npsi_coarse + s_coarse
                v_b_s = v_BS_coarse[b_s_coarse]  # v index
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_BS.append(v_b_s)
                # every other vertex is new
                s = 2 * s_coarse + 1
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                # bs_V[v_b_s] = b_s
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

            ###################################################
            # add every vertex to each new ring not in old mesh
            b = 2 * b_coarse + 1
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s in range(Npsi):
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

        self.F.append(F)
        self.v_BS.append(v_BS)
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))

    def write_plys(self, level=-1):
        if isinstance(level, int):
            vf_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_vf.ply"
            )
            he_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_he.ply"
            )
            print(f"Writing vertex-face ply to {vf_path}")
            self.v2h[level].write_source_ply(vf_path, use_ascii=False)
            print(f"Writing half-edge ply to {he_path}")
            self.v2h[level].write_target_ply(he_path, use_ascii=False)

        elif level == "all":
            for level in range(len(self.F)):
                self.write_plys(level=level)
            print(f"Done writing {self.name} plys.")

    def num_faces(self, p):
        """
        Nphi = 3 * 2**p
        Npsi = 2**p
        """
        return 6 * 4**p

    def num_vertices(self, p):
        return 3 * 4**p

    def num_edges(self, p):
        return 6 * 4**p + 3 * 4**p


class TorusFactory:
    """
    Makes and refines meshes for the tori,
    (sqrt(x**2 + y**2) - rad_big)**2 + z**2 - rad_small**2 = 0.


    Refinements are computed by doubling the number of




    Attributes
    ----------
    name : str
        Name of the mesh (name="torus")
    rad_big : float
        Big radius of the torus. 1 by default.
    rad_small : float
        Small radius of the torus. 1/3 by default.
    p0 : int
        Resolution parameter for lowest level of refinement.
    Nphi : list of int
        Number of azimuthal angle samples in each mesh. Controls resolution along the big circumference. 3 * 2**p by default.
    Npsi : list of int
        Number of samples along the small circumference. 2**p by default.
    ---
    ---
    xyz_coord_V : list of numpy.array
        xyz coordinates of the vertices in the finest mesh.
    F : list of list
        Face list for each mesh level.
    implicit_fun_str : str
        Implicit function for the torus.
    """

    @property
    def name(self):
        return self._name

    @property
    def rad_small(self):
        return self.rad_big / self.ratio_big2small

    def phi_of_xyz(self, x, y, z):
        return np.arctan2(y, x)

    def psi_of_xyz(self, x, y, z):
        return np.arctan2(z, np.sqrt(x**2 + y**2))

    def x_of_phi_psi(self, phi, psi):
        return np.cos(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    def y_of_phi_psi(self, phi, psi):
        return np.sin(phi) * (self.rad_big + np.cos(psi) * self.rad_small)

    def z_of_phi_psi(self, phi, psi):
        return np.sin(psi) * self.rad_small

    def Nphi(self, p):
        return self.ratio_big2small * 2**p

    def Npsi(self, p):
        return 2**p

    def __init__(self, p0=3, rad_big=1, ratio_big2small=3):
        self._name = "TESTtorus"
        self.p0 = p0
        self.rad_big = rad_big
        self.ratio_big2small = ratio_big2small
        rad_small = rad_big / ratio_big2small
        self.implicit_fun_str = (
            f"sqrt(x**2 + y**2) - {self.rad_big})**2 + z**2 - {rad_small}**2"
        )
        self.xyz_param_fun_str = [
            f"cos(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(psi) * {rad_small}",
        ]
        self.phipsi_param_fun_str = [
            f"cos(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(phi) * ({self.rad_big} + cos(psi) * {self.rad_small})",
            f"sin(psi) * {rad_small}",
        ]

        pow = self.p0
        N_small = 2**pow
        N_big = 3 * N_small

        xyz_coord_V = []
        F = []
        v_BS = []
        for b in range(N_big):
            phi = 2 * np.pi * b / N_big
            bp1 = (b + 1) % N_big
            for s in range(N_small):
                sp1 = (s + 1) % N_small
                phi_small = 2 * np.pi * s / N_small
                x = np.cos(phi) * (rad_big + np.cos(phi_small) * rad_small)
                y = np.sin(phi) * (rad_big + np.cos(phi_small) * rad_small)
                z = np.sin(phi_small) * rad_small
                xyz_coord_V.append(np.array([x, y, z]))
                b_s = b * N_small + s
                b_sp1 = b * N_small + sp1
                bp1_s = bp1 * N_small + s
                bp1_sp1 = bp1 * N_small + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_b_s = b_s
                v_BS.append(v_b_s)

        self.xyz_coord_V = xyz_coord_V
        self.F = [F]
        # self.pow = [pow]
        self.Nphi = [N_big]
        self.Npsi = [N_small]
        self.v_BS = [v_BS]
        self.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*self.VF())]

    @classmethod
    def build_test_plys(cls, num_refine=5):
        b = cls()
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")

    @property
    def pow(self):
        return [_ for _ in range(3, 3 + len(self.F))]

    def Vindices(self, level=-1):
        return self.v_BS[level]

    def num_vertices(self, level=-1):
        return len(self.Vindices(level))

    def num_faces(self, level=-1):
        return len(self.Vindices(level))

    def VF(self, level=-1):
        F = self.F[level]
        V = [self.xyz_coord_V[v] for v in self.Vindices(level)]
        return V, F

    def refine(self, convert_to_half_edge=True):
        r_b = self.rad_big
        r_s = self.rad_small
        Npsi_coarse = self.Npsi[-1]
        Nphi_coarse = self.Nphi[-1]
        pow_coarse = self.pow[-1]
        Npsi = 2 * Npsi_coarse
        Nphi = 2 * Nphi_coarse
        pow = pow_coarse + 1
        print(f"Refining {self.name}...")
        print(f"num_vertices: {Nphi_coarse*Npsi_coarse}-->{Nphi*Npsi}")
        self.Npsi.append(Npsi)
        self.Nphi.append(Nphi)
        self.pow.append(pow)
        F = []
        v_BS = []
        v_BS_coarse = self.v_BS[-1]

        for b_coarse in range(Nphi_coarse):
            ###################################################
            # add every other vertex to each ring in old mesh
            b = 2 * b_coarse
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s_coarse in range(Npsi_coarse):
                # every other vertex is the same as the coarse mesh
                s = 2 * s_coarse
                b_s_coarse = b_coarse * Npsi_coarse + s_coarse
                v_b_s = v_BS_coarse[b_s_coarse]  # v index
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])
                v_BS.append(v_b_s)
                # every other vertex is new
                s = 2 * s_coarse + 1
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                # bs_V[v_b_s] = b_s
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

            ###################################################
            # add every vertex to each new ring not in old mesh
            b = 2 * b_coarse + 1
            bp1 = (b + 1) % Nphi
            phi = 2 * np.pi * b / Nphi
            for s in range(Npsi):
                v_b_s = len(self.xyz_coord_V)  # new v index
                psi = 2 * np.pi * s / Npsi
                x = np.cos(phi) * (r_b + np.cos(psi) * r_s)
                y = np.sin(phi) * (r_b + np.cos(psi) * r_s)
                z = np.sin(psi) * r_s
                self.xyz_coord_V.append(np.array([x, y, z]))
                sp1 = (s + 1) % Npsi
                b_s = b * Npsi + s
                b_sp1 = b * Npsi + sp1
                bp1_s = bp1 * Npsi + s
                bp1_sp1 = bp1 * Npsi + sp1
                v_BS.append(v_b_s)
                F.append([b_s, bp1_sp1, bp1_s])
                F.append([b_s, b_sp1, bp1_sp1])

        self.F.append(F)
        self.v_BS.append(v_BS)
        if convert_to_half_edge:
            print("Converting to half-edge mesh...")
            self.v2h.append(VertTri2HalfEdgeConverter.from_source_samples(*self.VF()))

    def write_plys(self, level=-1):
        if isinstance(level, int):
            vf_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_vf.ply"
            )
            he_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_he.ply"
            )
            print(f"Writing vertex-face ply to {vf_path}")
            self.v2h[level].write_source_ply(vf_path, use_ascii=False)
            print(f"Writing half-edge ply to {he_path}")
            self.v2h[level].write_target_ply(he_path, use_ascii=False)

        elif level == "all":
            for level in range(len(self.F)):
                self.write_plys(level=level)
            print(f"Done writing {self.name} plys.")

    def num_faces(self, p):
        """
        Nphi = 3 * 2**p
        Npsi = 2**p
        """
        return 6 * 4**p

    def num_vertices(self, p):
        return 3 * 4**p

    def num_edges(self, p):
        return 6 * 4**p + 3 * 4**p


##################################################
##################################################
##################################################
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
        self.sample_dtypes = sample_dtypes
        self.count = len(samples)
        self.structured_array = np.array(samples, dtype=sample_dtypes)


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

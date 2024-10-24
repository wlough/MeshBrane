from plyfile import PlyData, PlyElement, PlyListProperty
from temp_python.src_python.half_edge_base_utils import (
    vf_samples_to_he_samples,
    he_samples_to_vf_samples,
    find_h_right_B,
)
from temp_python.src_python.utilities.misc_utils import (
    # make_output_dir,
    # load_npz,
    save_npz,
    # unchunk_file_with_cat,
)
import numpy as np

# _NUMPY_INT_ = np.int64
# _NUMPY_FLOAT_ = np.float64
# INT_TYPE = "int32"  # str(np.dtype(_NUMPY_INT_))
# FLOAT_TYPE = str(np.dtype(_NUMPY_FLOAT_))
from temp_python.src_python.global_vars import (
    INT_TYPE,
    FLOAT_TYPE,
    _NUMPY_INT_,
    _NUMPY_FLOAT_,
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
        # ply_format="binary_little_endian 1.0",
        comments=None,
        elements=None,
    ):
        self.name = name
        # self.ply_format = ply_format
        self.identifier = identifier
        self.elements = elements

    # def _generate_header(self):
    #     header = [self.identifier]
    #     header.append(f"format {self.ply_format}")
    #     for element in self.elements:
    #         header.append(f"element {element['name']} {element['count']}")
    #         for prop in element["properties"]:
    #             if len(prop) == 2:
    #                 header.append(f"property {prop[1]} {prop[0]}")
    #             elif len(prop) == 3:
    #                 header.append(f"property list uint8 {prop[1]} {prop[0]}")
    #     header.append("end_header")
    #     return header

    # def _print_header(self):
    #     for line in self.header:
    #         print(line)

    # @classmethod
    # def _schema_from_ply_file(cls, file_path, schema_name=None):
    #     """Constructs a PlySchema object by reading a ply file header. need to manually set size for list properties"""
    #     if schema_name is None:
    #         schema_name = file_path
    #     elements = []
    #     comments = []
    #     identifier = None
    #     ply_format = None
    #     with open(file_path, "r") as f:
    #         lines = f.readlines()
    #         identifier = lines[0].strip()
    #         i = 1
    #         while i < len(lines):
    #             line = lines[i].strip()
    #             if line.startswith("format"):
    #                 _, ply_format = line.split(maxsplit=1)
    #             elif line.startswith("comment"):
    #                 comments.append(line[8:].strip())
    #             elif line.startswith("element"):
    #                 elements.append(dict())
    #                 _, element_name, element_count = line.split()
    #                 elements[-1]["name"] = element_name
    #                 elements[-1]["count"] = element_count
    #                 elements[-1]["properties"] = []
    #             elif line.startswith("property"):
    #                 # prop_line = lines[i].strip()
    #                 prop_line_split = lines[i].strip().split()
    #                 if len(prop_line_split) == 3:
    #                     _, prop_dtype, prop_name = prop_line_split
    #                     elements[-1]["properties"].append((prop_name, prop_dtype))
    #                 if len(prop_line_split) == 5:
    #                     _, _, list_index_dtype, prop_dtype, prop_name = prop_line_split
    #                     elements[-1]["properties"].append((prop_name, prop_dtype, (3,)))
    #             elif line.startswith("end_header"):
    #                 break
    #             i += 1
    #     return cls(
    #         name=schema_name,
    #         identifier=identifier,
    #         ply_format=ply_format,
    #         comments=comments,
    #         elements=elements,
    #     )

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
            dtype=[("x", FLOAT_TYPE), ("y", FLOAT_TYPE), ("z", FLOAT_TYPE)],
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", INT_TYPE, (3,))])
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
        Data type for float properties (default=FLOAT_TYPE).
    int_type : str
        Data type for int properties (default=INT_TYPE).
    """

    def __init__(self):
        comments = [
            "Mesh represented by a list vertex positions and a list of vertex indices that make up each face"
        ]
        super().__init__(
            name="VertexTriMeshSchema",
            identifier="ply",
            # ply_format="binary_little_endian 1.0",
            comments=comments,
            # elements=None,
        )
        self.elements = [
            {
                "name": "vertex",
                "count": 7,
                "properties": [
                    ("x", FLOAT_TYPE),
                    ("y", FLOAT_TYPE),
                    ("z", FLOAT_TYPE),
                ],
            },
            {
                "name": "face",
                "count": 6,
                "properties": [("vertex_indices", INT_TYPE, (3,))],
            },
        ]

    def ply_data_to_samples(self, plydata):
        """Constructs a lists of data from a PlyData object using the schema"""
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=_NUMPY_FLOAT_,
        ).T
        vvv_of_F = np.array(
            [
                vertex_indices.tolist()
                for vertex_indices in plydata["face"]["vertex_indices"]
            ],
            dtype=_NUMPY_INT_,
        )
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
                ("x", FLOAT_TYPE),
                ("y", FLOAT_TYPE),
                ("z", FLOAT_TYPE),
            ],
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", INT_TYPE, (3,))])
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

    def __init__(self):
        identifier = "he_ply"
        comments = ["Schema for HalfEdgeMesh ply with boundary."]
        # super().__init__(name, format, elements)
        super().__init__(
            name="HalfEdgeMeshSchema",
            identifier="ply",
            # ply_format="binary_little_endian 1.0",
            comments=comments,
            # elements=elements,
        )
        self.elements = [
            {
                "name": "vertex",
                "count": "Nvertex",
                "properties": [
                    ("x", FLOAT_TYPE),
                    ("y", FLOAT_TYPE),
                    ("z", FLOAT_TYPE),
                    ("h", INT_TYPE),
                ],
            },
            {
                "name": "half_edge",
                "count": "Nhalf_edge",
                "properties": [
                    ("v", INT_TYPE),
                    ("f", INT_TYPE),
                    ("n", INT_TYPE),
                    ("t", INT_TYPE),
                ],
            },
            {
                "name": "face",
                "count": "Nface",
                "properties": [
                    ("h", INT_TYPE),
                ],
            },
            {
                "name": "boundary",
                "count": "Nboundary",
                "properties": [
                    ("h", INT_TYPE),
                ],
            },
        ]

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
        h_right_B : list of int
            h_right_B[b] = some half-edge in the boundary b which is right of the complement of the mesh
        """
        # ply_data = PlyData.read(file_path)
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=_NUMPY_FLOAT_,
        ).T
        h_out_V = np.array(plydata["vertex"]["h"], dtype=_NUMPY_INT_)
        v_origin_H = np.array(plydata["half_edge"]["v"], dtype=_NUMPY_INT_)
        h_next_H = np.array(plydata["half_edge"]["n"], dtype=_NUMPY_INT_)
        h_twin_H = np.array(plydata["half_edge"]["t"], dtype=_NUMPY_INT_)
        f_left_H = np.array(plydata["half_edge"]["f"], dtype=_NUMPY_INT_)
        h_bound_F = np.array(plydata["face"]["h"], dtype=_NUMPY_INT_)
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
            h_right_B = np.array(plydata["boundary"]["h"], dtype=_NUMPY_INT_)
            samples += (h_right_B,)
        except KeyError:
            h_right_B = find_h_right_B(
                xyz_coord_V,
                h_out_V,
                v_origin_H,
                h_next_H,
                h_twin_H,
                f_left_H,
                h_bound_F,
            )
            samples += (h_right_B,)
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
            [(x, y, z, h) for (x, y, z), h in zip(xyz_coord_V, h_out_V)],
            dtype=[
                ("x", FLOAT_TYPE),
                ("y", FLOAT_TYPE),
                ("z", FLOAT_TYPE),
                ("h", INT_TYPE),
            ],
        )
        H_data = np.array(
            [
                (v, n, t, f)
                for v, n, t, f in zip(v_origin_H, h_next_H, h_twin_H, f_left_H)
            ],
            dtype=[
                ("v", INT_TYPE),
                ("n", INT_TYPE),
                ("t", INT_TYPE),
                ("f", INT_TYPE),
            ],
        )
        F_data = np.array(h_bound_F, dtype=[("h", INT_TYPE)])
        # # ***
        # print(V_data)
        # print(type(V_data))
        # print(self.float_type)
        # print(self.int_type)
        # # ***
        vertex_element = PlyElement.describe(V_data, "vertex")
        half_edge_element = PlyElement.describe(H_data, "half_edge")
        face_element = PlyElement.describe(F_data, "face")
        if len(samples) == 8:
            h_right_B = samples[7]
            B_data = np.array(h_right_B, dtype=[("h", INT_TYPE)])
            boundary_element = PlyElement.describe(B_data, "boundary")
            return PlyData(
                [vertex_element, half_edge_element, face_element, boundary_element],
                text=not use_binary,
            )

        h_right_B = find_h_right_B(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
        B_data = np.array(h_right_B, dtype=[("h", INT_TYPE)])
        boundary_element = PlyElement.describe(B_data, "boundary")
        return PlyData(
            [vertex_element, half_edge_element, face_element, boundary_element],
            text=not use_binary,
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
        source_ply_schema = VertexTriMeshSchema()
        source_ply_data = source_ply_schema.samples_to_ply_data(*source_samples)
        return cls(source_ply_data)

    @classmethod
    def from_target_samples(cls, *target_samples):
        target_ply_schema = HalfEdgeMeshSchema()
        pc = cls(
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_samples = target_samples
        pc._target_ply_data = target_ply_schema.samples_to_ply_data(*target_samples)
        return pc

    @classmethod
    def from_target_ply(cls, target_ply_path, get_source_stuff=False):
        # target_ply_schema = HalfEdgeMeshSchema()
        pc = cls(
            source_ply_data=None,
            source_ply_path=None,
        )
        pc._target_ply_data = PlyData.read(target_ply_path)
        pc._target_samples = pc.target_ply_schema.ply_data_to_samples(
            pc.target_ply_data
        )
        if get_source_stuff:
            pc._source_samples = pc.target_samples_to_source_samples(*pc.target_samples)
            pc._source_ply_data = pc.source_ply_schema.samples_to_ply_data(
                *pc.source_samples
            )
        return pc

    # @classmethod
    # def update_he_ply_with_h_right_B(cls, old_ply_path, new_ply_path):
    #     c = cls.from_target_ply(old_ply_path)
    #     c.write_target_ply(target_path=new_ply_path, use_ascii=False)

    @classmethod
    def _update_he_plys_with_h_right_B(cls):
        old_ply_dir = "./data/ply/binary"
        new_ply_dir = "./data/half_edge_base/ply"
        misc = ["annulus.ply", "hex_patch.ply", "hex_sector.ply"]
        neovius = ["neovius.ply", "neovius_coarse.ply", "neovius_fine.ply"]
        dumbbell = ["dumbbell.ply", "dumbbell_coarse.ply", "dumbbell_fine.ply"]
        torus = [f"torus_{N:06d}_he.ply" for N in [192, 768, 3072, 12288, 49152]]
        unit_sphere = [
            f"unit_sphere_{N:06d}_he.ply"
            for N in [12, 42, 162, 642, 2562, 10242, 40962]
        ]
        ply_names = misc + neovius + dumbbell + torus + unit_sphere
        ply_names_new = misc + neovius + dumbbell
        ply_names_new = [_[:-4] + "_he.ply" for _ in ply_names_new]
        ply_names_new += torus + unit_sphere
        for ply_name, ply_name_new in zip(ply_names, ply_names_new):
            old_ply_path = f"{old_ply_dir}/{ply_name}"
            new_ply_path = f"{new_ply_dir}/{ply_name_new}"
            c = cls.from_target_ply(old_ply_path)
            c.write_target_ply(target_path=new_ply_path, use_ascii=False)

    @classmethod
    def _oblatify_the_spheres(cls, ratio=0.75):
        old_ply_dir = "./data/half_edge_base/ply"
        new_ply_dir = "./data/half_edge_base/ply"
        ply_names = [
            f"unit_sphere_{N:06d}_he.ply"
            for N in [12, 42, 162, 642, 2562, 10242, 40962]
        ]
        ply_names_new = [
            f"oblate_{N:06d}_he.ply" for N in [12, 42, 162, 642, 2562, 10242, 40962]
        ]
        # ply_names = [f"torus_{N:06d}_he.ply" for N in [192, 768, 3072, 12288, 49152]]
        # ply_names_new = [
        #     f"oblate_{N:06d}_he.ply" for N in [192, 768, 3072, 12288, 49152]
        # ]

        for ply_name, ply_name_new in zip(ply_names, ply_names_new):
            old_ply_path = f"{old_ply_dir}/{ply_name}"
            new_ply_path = f"{new_ply_dir}/{ply_name_new}"
            c0 = cls.from_target_ply(old_ply_path)
            xyz_coord_V = c0.target_samples[0]
            xyz_coord_V[:, 2] *= ratio
            target_samples = (xyz_coord_V, *c0.target_samples[1:])
            c = cls.from_target_samples(*target_samples)
            c.write_target_ply(target_path=new_ply_path, use_ascii=False)

    def source_samples_to_target_samples(self, *source_samples):
        (V, F) = source_samples
        target_samples = vf_samples_to_he_samples(V, F)
        return target_samples

    def target_samples_to_source_samples(self, *target_samples):
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = target_samples
        (xyz_coord_V, vvv_of_F) = he_samples_to_vf_samples(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        source_samples = (xyz_coord_V, vvv_of_F)
        return source_samples

    def write_half_edge_arrays(
        self, target_path=None, compressed=False, chunk=False, remove_unchunked=False
    ):

        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.target_samples
        arr = {
            "xyz_coord_V": xyz_coord_V,
            "h_out_V": h_out_V,
            "v_origin_H": v_origin_H,
            "h_next_H": h_next_H,
            "h_twin_H": h_twin_H,
            "f_left_H": f_left_H,
            "h_bound_F": h_bound_F,
            "h_right_B": h_right_B,
        }
        save_npz(
            arr,
            target_path,
            compressed=compressed,
            chunk=chunk,
            remove_unchunked=remove_unchunked,
        )


class MeshConverterBase:
    """
    Reading/writing ply files

    Attributes
    ----------
    vf_ply_path : str
        Path to the vf ply file where vf_ply_data was saved to/loaded from.
    vf_ply_data : PlyData
        Data from the vf ply file.
    vf_samples : tuple of ndarray
        (xyz_coord_V, V_of_F)
    he_ply_path : str
        Path to the he ply file where he_ply_data was saved to/loaded from.
    he_ply_data : PlyData
        Data from the source ply file.
    he_samples : tuple of ndarray
        (xyz_coord_V,..., h_right_B)

    Methods
    -------
    ...
    """

    # source_ply_schema = PlySchema()
    # target_ply_schema
    def __init__(self):
        self.vf_ply_path = None
        self.vf_ply_data = None
        self.vf_samples = None

        self.he_ply_path = None
        self.he_ply_data = None
        self.he_samples = None

    @classmethod
    def from_vf_ply(cls, ply_path, compute_he_stuff=True):
        c = cls()

        c.vf_ply_path = ply_path
        c.vf_ply_data = PlyData.read(ply_path)
        c.vf_samples = c.vf_ply_data_to_samples()

        if compute_he_stuff:
            c.he_samples = c.vf_samples_to_he_samples()
            c.he_ply_data = c.he_samples_to_ply_data()

        return c

    @classmethod
    def from_vf_samples(cls, xyz_coord_V, V_of_F, compute_he_stuff=True):
        c = cls()

        c.vf_samples = (xyz_coord_V, V_of_F)
        c.vf_ply_data = c.vf_samples_to_ply_data()

        if compute_he_stuff:
            c.he_samples = c.vf_samples_to_he_samples()
            c.he_ply_data = c.he_samples_to_ply_data()

        return c

    @classmethod
    def from_he_ply(cls, ply_path, compute_vf_stuff=True):
        c = cls()

        c.he_ply_path = ply_path
        c.he_ply_data = PlyData.read(ply_path)
        c.he_samples = c.he_ply_data_to_samples()

        if compute_vf_stuff:
            c.vf_samples = c.he_samples_to_vf_samples()
            c.vf_ply_data = c.vf_samples_to_ply_data()

        return c

    @classmethod
    def from_he_samples(
        cls,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
        compute_vf_stuff=True,
    ):
        c = cls()

        c.he_samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        c.he_ply_data = c.he_samples_to_ply_data()

        if compute_vf_stuff:
            c.vf_samples = c.he_samples_to_vf_samples()
            c.vf_ply_data = c.vf_samples_to_ply_data()

        return c

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

    def vf_samples_to_he_samples(self):
        (xyz_coord_V, V_of_F) = self.vf_samples
        he_samples = vf_samples_to_he_samples(xyz_coord_V, V_of_F)
        return he_samples

    def vf_ply_data_to_samples(self):
        """Constructs a lists of data from a PlyData object using the schema"""
        plydata = self.vf_ply_data
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=FLOAT_TYPE,
        ).T
        V_of_F = np.array(
            [
                vertex_indices.tolist()
                for vertex_indices in plydata["face"]["vertex_indices"]
            ],
            dtype=INT_TYPE,
        )
        samples = (
            xyz_coord_V,
            V_of_F,
        )
        return samples

    def vf_samples_to_ply_data(self, use_binary=True):
        """Constructs a PlyData object using the schema"""
        (
            xyz_coord_V,
            V_of_F,
        ) = self.vf_samples
        V_data = np.array(
            [tuple(v) for v in xyz_coord_V],
            dtype=[
                ("x", FLOAT_TYPE),
                ("y", FLOAT_TYPE),
                ("z", FLOAT_TYPE),
            ],
        )
        F_data = np.empty(len(V_of_F), dtype=[("vertex_indices", INT_TYPE, (3,))])
        F_data["vertex_indices"] = V_of_F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        return PlyData([vertex_element, face_element], text=not use_binary)

    def he_samples_to_vf_samples(self):
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.he_samples
        (xyz_coord_V, V_of_F) = he_samples_to_vf_samples(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        vf_samples = (xyz_coord_V, V_of_F)
        return vf_samples

    def he_ply_data_to_samples(self):
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
        h_right_B : list of int
            h_right_B[b] = some half-edge in the boundary b which is right of the complement of the mesh
        """
        plydata = self.he_ply_data
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=FLOAT_TYPE,
        ).T
        h_out_V = np.array(plydata["vertex"]["h"], dtype=INT_TYPE)
        v_origin_H = np.array(plydata["half_edge"]["v"], dtype=INT_TYPE)
        h_next_H = np.array(plydata["half_edge"]["n"], dtype=INT_TYPE)
        h_twin_H = np.array(plydata["half_edge"]["t"], dtype=INT_TYPE)
        f_left_H = np.array(plydata["half_edge"]["f"], dtype=INT_TYPE)
        h_bound_F = np.array(plydata["face"]["h"], dtype=INT_TYPE)
        h_right_B = np.array(plydata["boundary"]["h"], dtype=INT_TYPE)
        samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        return samples

    def he_samples_to_ply_data(self, use_binary=True):
        """Constructs a PlyData object using the schema"""

        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.he_samples
        V_data = np.array(
            [(x, y, z, h) for (x, y, z), h in zip(xyz_coord_V, h_out_V)],
            dtype=[
                ("x", FLOAT_TYPE),
                ("y", FLOAT_TYPE),
                ("z", FLOAT_TYPE),
                ("h", INT_TYPE),
            ],
        )
        H_data = np.array(
            [
                (v, n, t, f)
                for v, n, t, f in zip(v_origin_H, h_next_H, h_twin_H, f_left_H)
            ],
            dtype=[
                ("v", INT_TYPE),
                ("n", INT_TYPE),
                ("t", INT_TYPE),
                ("f", INT_TYPE),
            ],
        )
        F_data = np.array(h_bound_F, dtype=[("h", INT_TYPE)])
        B_data = np.array(h_right_B, dtype=[("h", INT_TYPE)])

        vertex_element = PlyElement.describe(V_data, "vertex")
        half_edge_element = PlyElement.describe(H_data, "half_edge")
        face_element = PlyElement.describe(F_data, "face")
        boundary_element = PlyElement.describe(B_data, "boundary")
        return PlyData(
            [vertex_element, half_edge_element, face_element, boundary_element],
            text=not use_binary,
        )

    def write_vf_ply(self, ply_path, use_binary=True):
        self.vf_ply_data.text = not use_binary
        self.vf_ply_data.write(ply_path)

    def write_he_ply(self, ply_path, use_binary=True):
        self.he_ply_data.text = not use_binary
        self.he_ply_data.write(ply_path)

    def write_he_samples(
        self, path=None, compressed=False, chunk=False, remove_unchunked=False
    ):

        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        ) = self.he_samples
        arr = {
            "xyz_coord_V": xyz_coord_V,
            "h_out_V": h_out_V,
            "v_origin_H": v_origin_H,
            "h_next_H": h_next_H,
            "h_twin_H": h_twin_H,
            "f_left_H": f_left_H,
            "h_bound_F": h_bound_F,
            "h_right_B": h_right_B,
        }
        save_npz(
            arr,
            path,
            compressed=compressed,
            chunk=chunk,
            remove_unchunked=remove_unchunked,
        )

    #####################################
    # for dealing with old data that doesn't include h_right_B
    def no_boundary_he_ply_data_to_samples(self):
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
        h_right_B : list of int
            h_right_B[b] = some half-edge in the boundary b which is right of the complement of the mesh
        """
        plydata = self.he_ply_data
        xyz_coord_V = np.array(
            [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]],
            dtype=FLOAT_TYPE,
        ).T
        h_out_V = np.array(plydata["vertex"]["h"], dtype=INT_TYPE)
        v_origin_H = np.array(plydata["half_edge"]["v"], dtype=INT_TYPE)
        h_next_H = np.array(plydata["half_edge"]["n"], dtype=INT_TYPE)
        h_twin_H = np.array(plydata["half_edge"]["t"], dtype=INT_TYPE)
        f_left_H = np.array(plydata["half_edge"]["f"], dtype=INT_TYPE)
        h_bound_F = np.array(plydata["face"]["h"], dtype=INT_TYPE)
        h_right_B = find_h_right_B(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
        samples = (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )
        return samples

    @classmethod
    def from_no_boundary_he_ply(cls, ply_path, compute_vf_stuff=True):
        c = cls()

        c.he_ply_path = ply_path
        c.he_ply_data = PlyData.read(ply_path)
        c.he_samples = c.no_boundary_he_ply_data_to_samples()
        c.he_ply_data = c.he_samples_to_ply_data()
        if compute_vf_stuff:
            c.vf_samples = c.he_samples_to_vf_samples()
            c.vf_ply_data = c.vf_samples_to_ply_data()

        return c

    @classmethod
    def update_no_boundary_he_plys(cls):
        old_ply_dir = "./data/ply/binary"
        new_ply_dir = "./data/half_edge_base/ply"
        misc = ["annulus.ply", "hex_patch.ply", "hex_sector.ply"]
        neovius = ["neovius.ply", "neovius_coarse.ply", "neovius_fine.ply"]
        dumbbell = ["dumbbell.ply", "dumbbell_coarse.ply", "dumbbell_fine.ply"]
        torus = []  # [f"torus_{N:06d}_he.ply" for N in [192, 768, 3072, 12288, 49152]]
        unit_sphere = [
            f"unit_sphere_{N:06d}_he.ply"
            for N in [12, 42, 162, 642, 2562, 10242, 40962]
        ]
        old_plys = misc + neovius + dumbbell + torus + unit_sphere
        he_plys = misc + neovius + dumbbell
        he_plys = [_[:-4] + "_he.ply" for _ in he_plys]
        he_plys += torus + unit_sphere
        vf_plys = [_[:-7] + "_vf.ply" for _ in he_plys]
        for old_ply, he_ply, vf_ply in zip(old_plys, he_plys, vf_plys):
            old_ply_path = f"{old_ply_dir}/{old_ply}"
            he_ply_path = f"{new_ply_dir}/{he_ply}"
            vf_ply_path = f"{new_ply_dir}/{vf_ply}"
            c = cls.from_no_boundary_he_ply(old_ply_path, compute_vf_stuff=True)
            c.write_he_ply(he_ply_path)
            c.write_vf_ply(vf_ply_path)

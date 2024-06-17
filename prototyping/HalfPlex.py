from typing import Tuple, List, Set
from itertools import combinations
from src.utilities import KeyManager, KeyCounter, parity_of_sort_permutation
import random

# just get up to EuclideanSpace and a method to add points that define corners of subdomain boxes in the hierearchy

# ___Space : ___Point container
#   -has a ___PointFactory that generates ___Points with unique sort_keys
#   -has a ___CellFactory that generates ___Cells with a parity
# ___Point : ___HalfPlexObject,
#   -has a unique integer-valued sort_key
# ___Simplex : ___Point container, ___Complex element
#   -___Points and possibely info about order/parity
# ___Complex : knows about ___Space, can ask ___Space for ___Simplex, contains ___Simplex


#################################################
# Inherit directly from MeshBraneObject #
#################################################
class MeshBraneObject:
    """
    Will be used to glue together C-GLASS and MeshBraneObject objects
    """

    FLAG_CONTAINER = 0b00001
    FLAG_ELEMENT = 0b00010

    def __init__(self):
        self.bit_flags = 0

    def set_container(self):
        self.bit_flags |= self.FLAG_CONTAINER

    def is_container(self):
        return bool(self.bit_flags & self.FLAG_CONTAINER)

    def clear_container(self):
        self.bit_flags &= ~self.FLAG_CONTAINER

    def set_element(self):
        self.bit_flags |= self.FLAG_ELEMENT

    def is_element(self):
        return bool(self.bit_flags & self.FLAG_ELEMENT)

    def clear_element(self):
        self.bit_flags &= ~self.FLAG_ELEMENT


class AbstractPoint(MeshBraneObject):
    """
    Base class for point-like things. Each instance of AbstractPoint may belong to multiple instances of AbstractSimplex, but only one instance of AbstractSpace.

    Properties
    ----------
    sort_key : int
        integer-valued key for sorting and fast comparison with other points in the same space

    Methods
    -------
    __eq__(self, other: object) -> bool
        two points are equal if they have the same sort_key and space
    __lt__(self, other: object) -> bool
        self.key < other.key
    __hash__(self) -> int
        return hash of (class_type, sort_key)
    """

    def __init__(self, sort_key, space):
        super().__init__()
        assert isinstance(sort_key, int), "sort_key must be an integer"
        self._sort_key = sort_key
        self._space = space

    @property
    def sort_key(self) -> int:
        return self._sort_key

    @property
    def space(self):
        return self._space

    def __hash__(self) -> int:
        return hash((type(self), self.sort_key))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.space == other.space and self.sort_key == other.sort_key
        return False

    def __lt__(self, other: object) -> bool:  # for sorting
        if isinstance(other, type(self)) and self.space == other.space:
            return self.key < other.key
        return NotImplemented


class AbstractSimplex(MeshBraneObject):
    """
    Base class for simplex-like things that contain AbstractPoint instances

    Properties
    ----------
    points : Set[AbstractPoint]
        Set of points that generate the simplex
    parity : bool
        orientation of the cell relative to the cell whose points are sorted by sort_key
    """

    def __init__(self, points, parity=None):
        super().__init__()
        self._points = frozenset(points)
        self._parity = parity

    @property
    def points(self) -> Set:
        return self._points

    @property
    def parity(self) -> bool:
        return self._parity

    def __eq__(self, other):
        return self.points == other.points and self.parity == other.parity

    def __hash__(self):
        return hash((self.points, self.parity))

    def __len__(self):
        return len(self.points)

    def __contains__(self, point):
        return point in self.points

    def __ge__(self, other):
        return self.points >= other.points

    def __gt__(self, other):
        return self.points > other.points

    def __le__(self, other):
        return self.points <= other.points

    def __lt__(self, other):
        return self.points < other.points


class AbstractFactory(MeshBraneObject):
    """
    Factory for creating instances a MeshBraneObject subclass. Each call to the factory creates a new object with constructor parameters determined by the tuple of factory_args passed to the factory.

    Properties
    ----------
    product_class : MeshBraneObject subclass
        class of objects created by the factory

    Methods
    -------
    __call__(self, *factory_args) -> MeshBraneObject
        Modify this method to define how constructor parameters for the product_class are obtained from factory_args.

    """

    def __init__(self, product_class):

        self._product_class = product_class

    def __call__(self, *factory_args):
        """
        Build tuple of constructor parameters for and return instance of product_class
        """
        product_args = ()
        return self.product_class(*product_args)

    @property
    def product_class(self):
        return self._product_class


#################################################
# Euclidean stuff #
#################################################
class EuclideanPoint(AbstractPoint):
    """
    A marked point in Euclidean space

    Properties
    ----------
    coordinates : List[float,...]
       xy- or xyz-coordinates of the point
    space : EuclideanSpace
        space the point lives in
    sort_key : int
        integer-valued key for sorting and fast comparison
    Methods
    -------
    __eq__(self, other: object) -> bool
        two points are equal if they have the same sort_key and space
    __lt__(self, other: object) -> bool
        self.key < other.key
    """

    def __init__(self, sort_key: int, space, coordinates: List[float]):
        super().__init__(sort_key, space)  # inherited from Object
        self.coordinates = coordinates  # list of cartesian coordinates

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        assert (
            isinstance(value, list) and len(value) == self.space.dimension
        ), "coordinates must be a list of length equal to the dimension of space"
        self._coordinates = value

    @property
    def space(self):
        return self._space


class EuclideanSimplex(AbstractSimplex):
    """
    Oriented Euclidean simplex with order 0,1,2,3

    Properties
    ----------
    points : Set[EuclideanPoint]
        Set of points that generate the simplex
    parity : bool
        orientation of the cell relative to the cell whose points are sorted by sort_key
    """

    def __init__(self, points, parity):
        super().__init__(points, parity)

    def points_list(self):
        sorted_points = sorted(self.points)
        if self.parity:
            return sorted_points
        else:
            sorted_points[0], sorted_points[1] = sorted_points[1], sorted_points[0]
            return sorted_points


class PointFactory(AbstractFactory):
    """
    Factory for creating instances of EuclideanPoint with given coordinates inside EuclideanSpace. Each call to the factory creates a new point with a unique key. Points are initialized with constructor parameters defined in __call__.

     Properties
     ----------
     product_class : MeshBraneObject object subclass
         class of objects created by the factory (EuclideanPoint)
     key_manager : KeyManager
         generate keys for objects created by the factory
     space : EuclideanSpace
         space the points live in

     Methods
     -------
     __call__(self, *factory_args) -> EuclideanPoint
         creates a single EuclideanPoint instance from factory_args
    """

    def __init__(self, space):
        super().__init__(EuclideanPoint)
        self._space = space
        self._key_manager = KeyManager()

    def __call__(self, *factory_args) -> Tuple:
        (coordinates,) = factory_args
        product_key = self.key_manager(self.product_class)
        product_args = (product_key, self.space, coordinates)
        return self.product_class(*product_args)

    @property
    def space(self):
        return self._space

    def create_points(self, *coordinates):
        """
        Create a list of points from a list of coordinates

        Parameters
        ----------
        coordinates: list of tuples of coordinates

        Returns
        -------
        list of EuclideanPoint
        """
        return [self(coordinates) for coordinates in coordinates]


class SimplexFactory(AbstractFactory):
    """
    Factory for creating instances of AbstractSimplex from sets of points in a EuclideanSpace.

    Properties
    ----------
    product_class : AbstractSimplex class
        class of objects created by the factory
    key_manager : KeyManager
        generates integer-valued key(s) for each instance of product_class created by the factory

    Methods
    -------
    __call__(self, *factory_args)
        create a new OrientedSimplex
    """

    def __init__(self, space):
        super().__init__(EuclideanSimplex)
        self._space = space
        self._key_manager = KeyManager()

    @property
    def space(self):
        return self._complex.space

    @property
    def key_manager(self):
        return self._key_manager

    def __call__(self, *factory_args):
        """
        Creates an OrientedSimplex from a list of points
        """
        (points_list,) = factory_args
        product_args = (frozenset(points_list), parity_of_sort_permutation(points_list))
        return self.product_class(*product_args)

    def with_faces(self, *factory_args):
        """
        Creates an OrientedSimplex from a list of points, creates OrientedSimplex of all faces of the simplex
        """
        (points_list,) = factory_args
        facet_parity = parity_of_sort_permutation(points_list)
        product_args_list = [(frozenset(points_list), parity)]

        product_args = (frozenset(points_list), parity_of_sort_permutation(points_list))
        return self.product_class(*product_args)


class EuclideanSpace(MeshBraneObject):
    """Acts as a container for labeled points in Euclidean space

    Attributes
    ----------
    dimension : int
        dimension of the space (2 or 3)
    points : Set[EuclideanPoint]
        current set of labeled points in the space
    point_factory : PointFactory
        factory for creating new points in the space

    """

    def __init__(self, dimension):
        super().__init__()
        self._dimension = dimension
        self._points = set()
        self._point_factory = PointFactory(self)
        # add origin and basis points at unit distance along each coordinate axis
        self._basis_points = [self.add_point([0.0] * dimension)]
        for i in range(self.dimension):
            self._basis_points.append(
                self.add_point(
                    ([1.0 if i == j else 0.0 for j in range(self.dimension)])
                )
            )

    @property
    def dimension(self):
        return self._dimension

    @property
    def points(self):
        return self._elements

    @property
    def point_factory(self):
        return self._point_factory

    @property
    def basis_points(self):
        return self._basis_points

    @property
    def origin(self):
        return self._basis_points[0]

    def __len__(self):
        return len(self.points)

    def __contains__(self, point):
        return point in self.points

    def __ge__(self, other):
        return self.points >= other.points

    def __gt__(self, other):
        return self.points > other.points

    def __le__(self, other):
        return self.points <= other.points

    def __lt__(self, other):
        return self.points < other.points

    def add_point(self, coordinates):
        """
        Create a EuclideanPoint with the given coordinates and add it to the self.points set

        Parameters
        ----------
            coordinates : tuple of floats
        """
        point = self.point_factory(coordinates)
        self.points.add(point)
        return point

    def yeet_point(self, point):
        """
        Remove a point from the space

        Parameters
        ----------
            point : EuclideanPoint
        """
        self.points.remove(point)

    def yeet_points(self, points):
        """
        Remove a set of points from the space

        Parameters
        ----------
            points : set of EuclideanPoints
        """
        self.points.difference_update(points)

    def add_points(self, *coordinates_list):
        """
        Create a EuclideanPoint with each coordinates in the coordinates_list and add to self.points set

        Parameters
        ----------
            coordinates : list of tuple of floats
        """
        self.points.update(self.point_factory.create_points(*coordinates_list))

    def random_populate(self, x_range, y_range, z_range, n_points):
        """
        Create n_points points in the space with coordinates in the ranges

        Parameters
        ----------
        x_range : tuple of floats
            (min, max) range for x-coordinate
        y_range : tuple of floats
            (min, max) range for y-coordinate
        z_range : tuple of floats
            (min, max) range for z-coordinate
        n_points : int
            number of points to create
        """
        for _ in range(n_points):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            z = random.uniform(*z_range)
            self.add_point([x, y, z])


###################################


class AbstractFactory(HalfPlexObject):
    """
    Factory for creating instances a HalfPlex class. Each call to the factory creates a new object with constructor parameters determined by the tuple of factory_args passed to the factory.

    Properties
    ----------
    product_class : HalfPlex object subclass
        class of objects created by the factory

    Methods
    -------
    __call__(self, *factory_args) -> HalfPlex object
        Modify this method to define how constructor parameters for the product class are obtained from factory_args.

    """

    def __init__(self, product_class):

        self._product_class = product_class
        self._key_manager = KeyManager()

    def __call__(self, *factory_args):
        """
        Build tuple of constructor parameters for and return instance of product_class
        """
        key_type = self.product_class
        product_key = self.key_manager(key_type)
        product_args = (product_key,)
        return self.product_class(*product_args)

    @property
    def product_class(self):
        return self._product_class

    @property
    def key_manager(self) -> KeyManager:
        return self._key_manager


class AbstractSpace(HalfPlexObject):
    """
    Container for labeled points in an AbstractSpace

    Attributes
    ----------
    points : Set[AbstractPoint]
        current set of labeled points in the space
    point_factory : AbstractFactory
        factory for creating new points in the space
    """

    def __init__(self, dimension):
        super().__init__()
        self._points = set()
        self._point_factory = AbstractFactory(0, AbstractPoint)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def points(self) -> Set[AbstractPoint]:
        return self._points

    @property
    def point_factory(self) -> AbstractFactory:
        return self._point_factory

    def add_point(self, coordinates):
        """
        Create a AbstractPoint with the given coordinates and add it to the self.points set

        Parameters
        ----------
            coordinates : tuple of floats
        """
        point = self.point_factory(coordinates)
        self.points.add(point)
        return point

    def yeet_point(self, point):
        """
        Remove a point from the space

        Parameters
        ----------
            point : AbstractPoint
        """
        self.points.remove(point)

    def add_points(self, *coordinates_list):
        """
        Create a AbstractPoint with each coordinates in the coordinates_list and add to self.points set

        Parameters
        ----------
            coordinates : list of tuple of floats
        """
        self.points.update(self.point_factory.create_points(*coordinates_list))

    def random_populate(self, x_range, y_range, z_range, n_points):
        """
        Create n_points points in the space with coordinates in the ranges

        Parameters
        ----------
        x_range : tuple of floats
            (min, max) range for x-coordinate
        y_range : tuple of floats
            (min, max) range for y-coordinate
        z_range : tuple of floats
            (min, max) range for z-coordinate
        n_points : int
            number of points to create
        """
        for _ in range(n_points):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            z = random.uniform(*z_range)
            self.add_point([x, y, z])


#################################################
#
#################################################

#################################################
# Simplical Complexes #


class OrientedSimplex(HalfPlexContainer):
    """
    Oriented Euclidean simplex with order 0,1,2,3

    -----------------------------
    Properties
    ----------
    points : Set[EuclideanPoint]
    parity : bool
        orientation of the simplex relative to the simplex whose points are sorted by EuclideanPoint hash key
    dim : int
        simplical dimension (len(points) - 1)
    face : OrientedSimplex
        (n-1)-simplex incident on the self with same orientation
    coface : OrientedSimplex
        (n+1)-simplex incident on the self with same orientation
    next : OrientedSimplex
        next simplex in the boundary of the coface
    twin : OrientedSimplex
        simplex with same points but opposite orientation
    """

    def __init__(
        self,
        key,
        points,
        parity,
    ):
        super().__init__(key)
        self._points = points
        self._dim = len(points) - 1
        self._parity = parity
        self._face = None
        self._coface = None
        self._twin = None
        self._next = None

    def __contains__(self, point):
        return point in self.points

    def __iter__(self):
        return iter(self.points)

    @property
    def parity(self) -> bool:
        return self._parity

    @property
    def dim(self) -> bool:
        return self._dim

    @property
    def points(self) -> Set[EuclideanPoint]:
        return self._points

    def is_face_of(self, other: object) -> bool:
        if isinstance(other, OrientedSimplex):
            return self.points.issubset(other.points)
        return False

    def is_coface_of(self, other: object) -> bool:
        if isinstance(other, OrientedSimplex):
            return other.points.issubset(self.points)
        return False

    def is_incidient_to(self, other: object) -> bool:
        if isinstance(other, OrientedSimplex):
            return self.is_face_of(other) or self.is_coface_of(other)
        return False

    def is_adjacent_to(self, other: object) -> bool:
        if isinstance(other, OrientedSimplex):
            return not self.points.isdisjoint(other.points)
        return False


class SimplexFactory(FactoryBase):
    """
    Factory for creating instances of OrientedSimplex from sets of points in a EuclideanSpace. Used to build a manifold OrientedComplex from a list of the sets of points which make up its highest order simplices (e.g. build a half-edge mesh from a list sets of points in faces)

    Properties
    ----------
    key : int
        integer-valued key assigned to the factory
    product_class : HalfPlex object class
        class of objects created by the factory
    key_manager : KeyManager
        generates integer-valued key(s) for each instance of product_class created by the factory

    Methods
    -------
    __call__(self, *factory_args)
        create a new OrientedSimplex
    """

    def __init__(self, key, complex):
        super().__init__(key, OrientedSimplex)
        self._complex = complex

    def __call__(self, *factory_args):
        """
        Creates an OrientedSimplex and all of its OrientedSimplex faces from a list of points
        """
        (points,) = factory_args
        while len(self.key_manager) <= len(points):
            self._key_manager.append(KeyManager())
        product_key = self.key_manager[len(points)]()
        parity = self.parity_of_sort_permutation(points)
        product_args = (product_key, points, parity)
        return self.product_class(*product_args)

    @property
    def complex(self):
        return self._complex

    @property
    def space(self):
        return self._complex.space

    @property
    def key_manager(self):
        return self._key_manager

    def parity_of_sort_permutation(self, points):
        """
        Parity of the permutation requied to sort points by hash key

        Parameters
        ----------
        points : list of EuclideanPoint

        Returns
        -------
        bool : True for even permutation, False for odd for false
        """
        # V[i] = the index of the point in points with i-th largest hash key
        V = sorted(range(len(points)), key=points.__getitem__)
        visited = [False] * len(V)
        result = True
        for start in range(len(V)):
            if visited[start]:
                continue
            visited[start] = True
            j = V[start]
            while j != start:
                result = not result
                visited[j] = True
                j = V[j]
        return result

    def some_face_of(sup_simplex):
        # P = sorted(list(SupSimplex.points))
        if sup_simplex.parity:
            return cls(sorted(list(SupSimplex.points))[:-1])
        else:
            P = sorted(list(SupSimplex.points))
            return cls(P[:-2] + [P[-1]])


class OrientedComplex:
    """
    Container for a set of OrientedSimplex instances
    """

    def __init__(self, key, space):
        super().__init__(key)
        self._space = space
        self._simplex_factory = SimplexFactory(self)
        self._simplices = [set() for _ in range(space.dimension + 1)]

    @property
    def space(self):
        return self._space

    @property
    def simplex_factory(self):
        return self._simplex_factory

    @property
    def simplices(self):
        return self._simplices

    def add_simplex(self, points):
        """
        Create a OrientedSimplex with the given points list

        Parameters
        ----------
            points : set of EuclideanPoint
        """
        simplex = self.simplex_factory(points)
        while len(self.simplices) < simplex.dim + 1:
            self._simplices.append(set())
        self._simplices[simplex.dim].add(simplex)
        return simplex


####################################################################
# Simulations
####################################################################
class SimulationBase:
    """
    Base class for simulations

    Output parameters
    -----------------
    run_name : str
        name of the simulation run
    output_directory : str
        directory to save simulation state and frames
    Tsave_state : float
        time interval between saving simulation state
    Tsave_frame : float
        time interval between saving simulation frame

    Computational parameters
    ------------------------
    Tstep : float
        time interval between simulation steps
    Tend : float
        end time of the simulation

    Physical parameters
    -------------------

    """

    def __init__(self, space, complex_dict):
        self._space = space
        self._complex_dict = complex_dict


class SimulationInitializer:
    """
    Load simulation state from file or generate initial state
    """

    def __init__(self):
        pass


class ClosedMitosisSim(SimulationBase):
    """
    Simulation of closed mitosis

    nuclear envelope -- helfrich elastic membrane, 2d complex (manifold)
    spindle pole bodies -- rigid body, 3d complex with subset of faces adjacent to membrane patch
    microtubles -- semi-flexible filaments anchored to vertices of the SPB
    chromosomes --
    membrane proteins --
    """

    def __init__(self):
        super().__init__()


class MicrotublePropagationSim(SimulationBase):
    """
    Simulation of Meredith's microtuble model with signal propagation

    microtubles -- 2d complex homeomorphic to a cylinder
    embedded proteins --
    """

    def __init__(self):
        super().__init__()

# from src.HalfPlex import EuclideanSpace, OrientedSimplex, OrientedComplex, HalfPlexPoint
# from src.EuclideanSpaceViewer import SpaceViewer
from itertools import combinations,permutations
from src.utilities import parity_of_sort_permutation





def relative_parity(original_list, sublist):
    indices = [original_list.index(x) for x in sublist]
    parity = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                parity += 1
    return parity % 2
S = [0,1,2,3]
parity_facet=parity_of_sort_permutation(S)
SS = [(list(combo), True)  for k in range(len(S), 0, -1) for combo in combinations(S, k)]
SS = [(list(combo), relative_parity(S, combo))  for k in range(len(S), 0, -1) for combo in combinations(S, k)]


s = [3,2,1]
PP = sorted(s, key=lambda sk: S.index(sk))
J = sorted(range(len(U)), key=U.__getitem__)

pts_parity=
# Test
print(even_permutations([1, 2, 3]))
# %%
bdryS_parity = []
# %%
def get_bdry(S, S_parity):
    bdryS_parity = []
    for i in range(len(S)):
        s=[]
        parity = not parity
        for j in range(len(S)):
            if j != i:
                s.append(S[j])
        bdryS_parity.append((s, parity))
    return bdryS_parity
# %%
from itertools import combinations

def relative_parity(original_list, sublist):
    indices = [original_list.index(x) for x in sublist]
    parity = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                parity += 1
    return parity % 2

def faces_of_simplex(vertices):
    faces = list(combinations(vertices, len(vertices) - 1))
    return [face for face in faces if relative_parity(vertices, face) == 0]

# Test
vertices = ['v0', 'v1', 'v2', 'v3']  # Vertices of a 3-simplex
print(faces_of_simplex(vertices))
# %%
import random
class test:
    """
    Attributes
    ----------
    P : set of things
    N : number of things in P

    Mehtods
    -------
    listpop : return subset of P with N-1 elements by making list, popping element of list, and converting back to set
    setpop : return subset of P with N-1 elements by copying set and popping element of set
    """
    def __init__(self, P):
        self.P = P.copy()
        self.N = len(P)

    def listpop(self):
        P_list = list(self.P)
        P_list.pop()
        return set(P_list)

    def setpop(self):
        P = self.P.copy()
        P.pop()
        return P

N=111
P = set([random.randint(0, 333) for _ in range(N)])
t1 = test(P)
t2 = test(P)

%timeit t1.listpop()
%timeit t2.setpop()
# %%


E = EuclideanSpace(3)
E.random_populate(*3 * [[-1, 1]], 333)
S = OrientedComplex(0, E)

P = list(E.points)
p0, p1, p2, p3, p4, p5, p6 = list(E.points)[:7]
_V0, _V1, _V2, _V3, _V4 = [p0], [p1], [p2], [p3], [p4]
_E01, _E12, _E20, _E54 = [p0, p1], [p1, p2], [p2, p0], [p5, p4]
_F012, _F654 = [p0, p1, p2], [p6, p5, p4]
_C = [_V0, _V1, _V2, _V3, _V4, _E01, _E12, _E20, _E54, _F012, _F654]
C = [S.add_simplex(c) for c in _C]
[c.order for c in C]
S.simplices


#
# %%
import random

U = random.sample(P, 13)
I = [_ for _ in range(len(U))]
V = sorted(U)
J = [U.index(v) for v in V]
J2 = sorted(range(len(U)), key=U.__getitem__)


def parity_of_sort_by_cycles(U):
    """
    Determine parity of a permutation which sorts the list U using a cycle detection algorithm.

    Parameters
    ----------
    U : list of objects which have __lt__() method

    Returns
    -------
    bool : True for even permutation, False for odd for false

    """
    # If V=sorted(U), then J=[U.index(v) for v in V]
    J = sorted(range(len(U)), key=U.__getitem__)
    visited = [False] * len(U)
    parity = True
    for i in range(len(U)):
        if visited[i]:
            continue
        visited[i] = True
        j = J[i]
        while j != i:
            parity = not parity
            visited[j] = True
            j = J[j]
    return parity


def parity_of_sort_permutation(U):
    """
    Uses a cycle detection algorithm to determine the parity of the permutation required to sort a list of objects which have an __lt__ method

    Parameters
    ----------
    U : list of objects to be sorted

    Returns
    -------
    bool : True for even permutation, False for odd for false
    """
    # V[i] = the index of the point in points with i-th largest sort_key
    # sort iterable range(len(U))
    V = sorted(range(len(U)), key=U.__getitem__)
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


parity_of_sort_by_cycles(U) == parity_of_sort_permutation(U)
# %%
I = [_ for _ in range(333)]
Np = 13
U0 = random.sample(I, Np)
U0.sort()
U = U0.copy()
random.shuffle(U)
# %%

# %%
V = sorted(range(len(U)), key=U.__getitem__)
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


# %%
def perm(U, iters=1):
    V = [_ for _ in range(len(U))]
    for iter in range(iters):
        i, j = random.sample(V, 2)
        U[i], U[j] = U[j], U[i]
    return U


def parity_of_sort(U):
    """
    Parity of the permutation required to sort list of objects with an __lt__ method

    Parameters
    ----------
    U : list of objects to be sorted

    Returns
    -------
    bool : True for even permutation, False for odd for false
    """
    # V[i] = the index of the point in points with i-th largest sort_key
    V = sorted(range(len(U)), key=U.__getitem__)
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


class CustomIterable:
    def __init__(self, U):
        self.X = range(len(U))

    def __getitem__(self, i):
        return U[i]


def other_sort(U):
    J = CustomIterable(U)
    V = sorted(J)
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


parity_of_sort(U)
# for iter in range(10):
#
#     perm(U)
#     if parity_of_sort(U):
#         print("even")
#     else:
#         print("odd")
# %%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# %%
I = [_ for _ in range(111)]
X = [tuple(np.random.rand(3) * 0.6) for i in I]
Points = [Point(x, i) for i, x in zip(I, X)]
p0, p1, p2, p3, p4, p5, p6 = Points[:7]
_V0, _V1, _V2, _V3, _V4 = [p0], [p1], [p2], [p3], [p4]
_E01, _E12, _E20, _E54 = [p0, p1], [p1, p2], [p2, p0], [p5, p4]
_F012, _F654 = [p0, p1, p2], [p6, p5, p4]
V0, V1, V2 = (OrientedSimplex(_) for _ in (_V0, _V1, _V2))
E01, E12, E20, E54 = (OrientedSimplex(_) for _ in (_E01, _E12, _E20, _E54))
F012, F654 = (OrientedSimplex(_) for _ in (_F012, _F654))
someE = OrientedSimplex.some_order_n_minus_one_face(F012)
E01.some_order_n_minus_one_face_key1()


# %%

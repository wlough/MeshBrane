class HalfEdgeLaplacian(mesh, **kwargs):
    def __init__(self, mesh, **kwargs):
        super(HalfEdgeLaplacian, self).__init__(mesh, **kwargs)
        self.mesh = mesh
        self.build()

    def build(self):
        self.L = self.mesh.build_laplacian()

    def apply(self, x):
        return self.L.dot(x - self.mesh.boundary_project(x))


class ApplyToTol:
    def __init__(self, operator, mesh, tol=1e-6):
        self.operator = operator
        self.mesh = mesh
        self.tol = tol

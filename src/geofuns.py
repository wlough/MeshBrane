
def areal_coords(p, r0, r1, r2):
    """areal coordinates of p wrt triangle r0-r1-r2"""
    s = np.zeros(3)
    A01 = jitcross(r0, r1)
    A12 = jitcross(r1, r2)
    A20 = jitcross(r2, r0)

    Ap1 = jitcross(p, r1)
    # A12 = jitcross(r1, r2)
    A2p = jitcross(r2, p)

    A0p = jitcross(r0, p)
    Ap2 = jitcross(p, r2)
    # A20 = jitcross(r2, r0)

    # A01 = jitcross(r0, r1)
    A1p = jitcross(r1, p)
    Ap0 = jitcross(p, r0)

    A012 = A01 + A12 + A20
    normsqrA012 = A012[0] ** 2 + A012[1] ** 2 + A012[2] ** 2
    Ap12 = Ap1 + A12 + A2p
    A0p2 = A0p + Ap2 + A20
    A01p = A01 + A1p + Ap0
    s[0] = jitdot(A012, Ap12) / normsqrA012
    s[1] = jitdot(A012, A0p2) / normsqrA012
    s[2] = jitdot(A012, A01p) / normsqrA012
    return s

def barcell(self, v):
    """returns vertex positions of barycentric cell dual to v
    midpoints of edges and barycenters of faces"""
    V = []
    x0, y0, z0 = self.vertices[v]
    h = self.V_hedge[v]
    h_start = h
    while True:
        v1 = self.H_vertex[h]
        x1, y1, z1 = self.vertices[v1]
        h = self.H_prev[h]
        h = self.H_twin[h]
        v2 = self.H_vertex[h]
        x2, y2, z2 = self.vertices[v2]
        V.append([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        V.append([(x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3, (z0 + z1 + z2) / 3])

        if h == h_start:
            break
    return np.array(V)

def vorcell(self, v):
    """returns vertex positions of voroni cell dual to v"""
    V = []
    r0 = self.vertices[v]
    h = self.V_hedge[v]
    h_start = h
    while True:
        v1 = self.H_vertex[h]
        r1 = self.vertices[v1]
        h = self.H_prev[h]
        h = self.H_twin[h]
        v2 = self.H_vertex[h]
        r2 = self.vertices[v2]
        r01 = r1 - r0
        r12 = r2 - r1
        r20 = r0 - r2
        normsqr_r12 = r12[0] ** 2 + r12[1] ** 2 + r12[2] ** 2
        normsqr_r20 = r20[0] ** 2 + r20[1] ** 2 + r20[2] ** 2
        normsqr_r01 = r01[0] ** 2 + r01[1] ** 2 + r01[2] ** 2
        #######################
        c0 = normsqr_r12 * (normsqr_r20 + normsqr_r01 - normsqr_r12)
        c1 = normsqr_r20 * (normsqr_r01 + normsqr_r12 - normsqr_r20)
        c2 = normsqr_r01 * (normsqr_r12 + normsqr_r20 - normsqr_r01)
        c012 = c0 + c1 + c2
        c0 /= c012
        c1 /= c012
        c2 /= c012
        #######################
        # r01_x_r12 = jitcross(r01, r12)
        # normsqr_r01_x_r12 = (
        #     r01_x_r12[0] ** 2 + r01_x_r12[1] ** 2 + r01_x_r12[2] ** 2
        # )
        # r01_dot_r20 = jitdot(r01, r20)
        # r01_dot_r12 = jitdot(r01, r12)
        # r20_dot_r12 = jitdot(r20, r12)
        # c0 = -normsqr_r12 * r01_dot_r20 / (2 * normsqr_r01_x_r12)
        # c1 = -normsqr_r20 * r01_dot_r12 / (2 * normsqr_r01_x_r12)
        # c2 = -normsqr_r01 * r20_dot_r12 / (2 * normsqr_r01_x_r12)
        #######################
        midpoint01 = (r1 + r0) / 2
        circumcenter = c0 * r0 + c1 * r1 + c2 * r2

        V.append([*midpoint01])
        V.append([*circumcenter])

        if h == h_start:
            break
    return np.array(V)

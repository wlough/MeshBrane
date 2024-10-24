def get_index_of_twin(H, h):
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


def source_samples_to_target_samples(V, F):
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
            h_twin = get_index_of_twin(H, h)  # returns -1 if twin not found
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

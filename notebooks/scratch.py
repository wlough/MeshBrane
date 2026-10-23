import numpy as np
import matplotlib.pyplot as plt

Ne = 4
Nh = 8


# Ne =4, Nh=8
#
# .<-h7--.<-h6--.<-h5--.<-h4--.
# .--h0->.--h1->.--h2->.--h3->.--h4->.--h5->.--h6->.--h7->

h_twin_H0 = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=int)
h_rot_H0 = np.array([0, 7, 6, 5, 4, 3, 2, 1], dtype=int)

h_twin_H = np.zeros(Nh, dtype=int)
h_rot_H = np.zeros(Nh, dtype=int)

for h in range(Nh):
    ht = Nh - 1 - h
    ht = -(h + 1) % Nh
    hr = (Nh - h) % Nh
    print(f"{h,ht,hr=}")
    h_twin_H[h] = ht
    h_rot_H[h] = hr


h_twin_H - h_twin_H0
h_rot_H - h_rot_H0
# %%

Ne = 3
Nh = 6

#
# .<-h5--.<-h4--.<-h3--.
# .--h0->.--h1->.--h2->.

h_twin_H0 = np.array([5, 4, 3, 2, 1, 0], dtype=int)
h_rot_H0 = np.array([3, 5, 4, 0, 2, 1], dtype=int)
h_next_H0 = np.array([1, 2, 0, 5, 4, 3], dtype=int)

h_twin_H = np.zeros(Nh, dtype=int)
h_rot_H = np.zeros(Nh, dtype=int)
h_next_H = np.zeros(Nh, dtype=int)


for h in range(Ne):
    # ht = -(h + 1) % Nh
    ht = Nh - (h + 1)
    hn = (h + 1) % Ne
    h_twin_H[h] = ht
    h_twin_H[ht] = h
    h_next_H[h] = hn

for h in range(Ne, Nh):
    h_reduced = h - Ne
    hn_reduced = (h_reduced + 1) % Ne
    hn = hn_reduced + Ne
    h_next_H[h] = hn

for h in range(Nh):
    hr = h_next_H[h_twin_H[h]]
    h_rot_H[h] = hr

# for h in range(Nh):
#     ht = -(h + 1+Nh) % Nh
#     hr = (Nh - h) % Nh
#     print(f"{h,ht,hr=}")
#     h_twin_H[h] = ht
#     h_rot_H[h] = hr

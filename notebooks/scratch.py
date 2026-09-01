import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(0, 2 * np.pi, 250)
x0 = np.cos(phi)
y0 = np.sin(phi)


x_squish = 0.6
y_squish = 0.8
neck_radius = 0.2
c = 1 - neck_radius

# raw_point_cloud[:, 0] *= x_squish * (
#     1 - c * np.cos(np.pi * raw_point_cloud[:, 2] / 2)
# )  # Squish along x-axis
# raw_point_cloud[:, 1] *= y_squish * (
#     1 - c * np.cos(np.pi * raw_point_cloud[:, 2] / 2)
# )  # Squish along y-axis
a = 0.5
x = x0
y = y0


fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(3.25, 3.25),
)

ax.plot(x, y)
ax.set_aspect("equal")

plt.show()

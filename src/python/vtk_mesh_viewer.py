import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from mayavi import mlab


def shplot_mayavi():
    # Degree and order
    l = 1  # degree
    m = 0  # order

    # Visual appearance
    bump_height = 0.25
    ref_sphere = 0.0

    tt = np.linspace(0, np.pi, 41)
    pp = np.linspace(0, 2 * np.pi, 81)

    phi, theta = np.meshgrid(pp, tt)  # Define the mesh

    # Compute the spherical harmonic Y_l^m
    Ylm = sph_harm(m, l, phi, theta).real

    # Normalize entries to interval [-1.0, 1.0]
    maxYLM = np.max(np.abs(Ylm))
    Ylm = Ylm / maxYLM

    # Make radius <= 1 an affine mapping of the spherical harmonic
    radius = np.abs(ref_sphere + bump_height * Ylm) / (ref_sphere + bump_height)

    # Convert to 3D Cartesian mesh
    rsint = radius * np.sin(theta)
    x = rsint * np.cos(phi)
    y = rsint * np.sin(phi)
    z = radius * np.cos(theta)

    # Plotting with Mayavi
    mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
    surf = mlab.mesh(x, y, z, scalars=Ylm, colormap="viridis")

    # Add colorbar
    mlab.colorbar(surf, title="Ylm", orientation="vertical")

    # Add text
    Ylmstr = f"Re(Y_{l}^{m}(theta,phi))"
    mlab.text(0.65, 0.8, Ylmstr, width=0.2)

    # Set axis limits and turn off the axis
    mlab.axes(xlabel="X", ylabel="Y", zlabel="Z", ranges=[-1, 1, -1, 1, -1, 1])
    mlab.outline()

    # Save to PNG file
    mlab.savefig("./output/shplot.png")

    mlab.show()


def shplot_matplotlib():
    # Degree and order
    l = 6  # degree
    m = 1  # order

    # Visual appearance
    bump_height = 0.25
    ref_sphere = 1.0

    tt = np.linspace(0, np.pi, 41)
    pp = np.linspace(0, 2 * np.pi, 81)

    phi, theta = np.meshgrid(pp, tt)  # Define the mesh

    # Compute the spherical harmonic Y_l^m
    Ylm = sph_harm(m, l, phi, theta).real

    # Normalize entries to interval [-1.0, 1.0]
    maxYLM = np.max(np.abs(Ylm))
    Ylm = Ylm / maxYLM

    # Make radius <= 1 an affine mapping of the spherical harmonic
    radius = np.abs(ref_sphere + bump_height * Ylm) / (ref_sphere + bump_height)

    # Convert to 3D Cartesian mesh
    rsint = radius * np.sin(theta)
    x = rsint * np.cos(phi)
    y = rsint * np.sin(phi)
    z = radius * np.cos(theta)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(Ylm), linewidth=0.1)

    # Add lights and set viewpoint
    ax.view_init(elev=30, azim=40)

    # Set axis limits and turn off the axis
    maxa = 1.0
    ax.set_xlim([-maxa, maxa])
    ax.set_ylim([-maxa, maxa])
    ax.set_zlim([-maxa, maxa])
    ax.axis("off")

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(Ylm)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["$-1$", "$-\\frac{1}{2}$", "$0$", "$\\frac{1}{2}$", "$1$"])
    cbar.ax.tick_params(labelsize=12)

    # Add text
    Ylmstr = f"$\\Re\\left(Y_{{{l}}}^{{{m}}}(\\theta,\\phi)\\right)$"
    ax.text2D(0.65, 0.8, Ylmstr, fontsize=20, transform=ax.transAxes)

    # Save to PNG file
    plt.savefig("./output/shplot.png", dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    shplot_mayavi()

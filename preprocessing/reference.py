import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

N = 500


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def visualize_sphere():
    coordinates = load_reference_points()

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'}, figsize=(10, 10))
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], s=10, c='r', zorder=10)
    plt.show()


if __name__ == "__main__":
    points = sample_spherical(N).transpose(1, 0)

    with open("reference_directions.txt", "w") as file:
        for point in points:
            file.write("%.3f %.3f %.3f\n" % (point[0], point[1], point[2]))

    visualize_sphere()

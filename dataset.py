import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D


def load_itk(filename):
    itkimage = sitk.ReadImage(filename)

    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.transpose(2, 1, 0)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def load_points(path):
    with open(path, "r") as file:
        lines = file.readlines()

    points = [[float(value) for value in line.split(" ")] for line in lines]
    return np.array(points)


def generate_figure(idx):
    scan, origin, spacing = load_itk("./training/dataset0%d/image0%d.mhd" % (idx, idx))

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    for image_id in range(4):
        path = "./training/dataset0%d/vessel%s/reference.txt" % (idx, str(image_id))
        print(path)
        points = load_points(path)
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        X, Y, Z = X / spacing[1], Y / spacing[1], Z / spacing[0]
        ax.scatter(X, Y, Z, s=0.3, c=np.arange(len(points)))

    plt.axis("square")
    plt.savefig("./visualizations/" + str(idx) + ".png")


if __name__ == "__main__":
    for i in range(8):
        print("\nplotting dataset" + str(i))
        generate_figure(i)

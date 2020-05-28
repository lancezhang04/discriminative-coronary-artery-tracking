from model import create_model
import tensorflow as tf
import numpy as np
import SimpleITK
import matplotlib.pyplot as plt

W, D = 19, 500

"""
this script contains a variety of utility functions with different purposes
they are used to load the dataset, test the model, or preprocess the data for training
"""


def plot_slice(slice_):
    plt.figure(figsize=(5, 5))
    plt.imshow(slice_, cmap="gray")
    plt.axis("off")
    plt.show()


def load_itk(filename):
    itkimage = SimpleITK.ReadImage(filename)

    ct_scan = SimpleITK.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.transpose(2, 1, 0)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def load_points(path):
    with open(path, "r") as file:
        lines = file.readlines()

    points = [[float(value) for value in line.split(" ")] for line in lines]
    return np.array(points)[:, :4]


def create_fake_images(N=10, w=W):
    return tf.random.normal((N, w, w, w, 1), dtype=tf.dtypes.float32)


def create_fake_labels(r=1, d=D, N=10, on=0):
    prob = np.zeros([N, d], dtype="float32")
    prob[:, 0] = 1
    return [np.full(fill_value=on, shape=(N, 1)), np.full(fill_value=r, shape=(N, 1), dtype="float32"), prob]


def calculate_distance(point1, point2):
    if (type(point1) != np.ndarray) or (type(point2) != np.ndarray):
        point1, point2 = np.array(point1), np.array(point2)
    distance = (point1 - point2) ** 2
    distance = np.sqrt(np.sum(distance))
    return distance


def segment_image(image, point, w=W):
    """
    extract a cube from a given image that centers around a given point
    returns None if the cube extends outside of the given image

    :param image: the image (CCTA) to extract cube from
    :param point: the point at the center of the returned cube
    :param w: lengths of the sides of the cube
    :return: the extracted cube (3D array) or None
    """
    x, y, z = point
    w = int((w - 1) / 2)
    print(x - w, x + w + 1, y - w, y + w + 1)
    try:
        return image[x - w:x + w + 1, y - w:y + w + 1, z - w:z + w + 1]
    except:
        return None


if __name__ == "__main__":
    a = np.arange(8 * 64).reshape(8, 8, 8)
    print(segment_image(a, [20, 20, 20]).shape)

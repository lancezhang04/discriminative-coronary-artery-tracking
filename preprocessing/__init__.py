import sys
import os
sys.path.append(os.path.abspath(".."))

from warnings import filterwarnings
filterwarnings("ignore")

from utils import load_itk, calculate_distance
from scipy import ndimage
from numpy import savez_compressed
from preprocessing.reference import *


voxel_spacing = 0.5
D = 500


def load_reference_points(fn="reference_directions.txt"):
    """
    load the reference points previously generated and return them as a numpy array with shape (N, 3)
    """

    with open(fn, "r") as file:
        lines = file.readlines()

    coordinates = [[float(i) for i in line.strip().split(" ")] for line in lines]
    coordinates = np.array(coordinates)
    return coordinates


def find_direction_cat(point1, point2, reference_points):
    point2 -= point1
    point2 /= np.linalg.norm(point2)

    distances = []
    for r_point in reference_points:
        distances.append(angle_between(point2, r_point))
    distances = np.array(distances)

    return np.argmin(distances)


def create_sample(idx, points, reference_points):
    radius = points[idx, 3]
    point = points[idx, :3]
    directions_label = np.zeros(D, dtype="float32")

    for i in range(50, 201):
        assert idx + i < points.shape[0] - 1
        distance = calculate_distance(point, points[idx + i, :3])
        if distance > radius:
            print("forward: %.5f" % distance)
            forward_idx = idx + i
            break
    forward_d = find_direction_cat(point, points[forward_idx, :3], reference_points)
    directions_label[forward_d] = 0.5

    for i in range(50, 201):
        assert idx - i > 0
        distance = calculate_distance(point, points[idx - i, :3])
        if distance > radius:
            print("backward: %.5f" % distance)
            backward_idx = idx - i
            break
    backward_d = find_direction_cat(point, points[backward_idx, :3], reference_points)
    directions_label[backward_d] = 0.5

    return radius, directions_label


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def change_voxel_spacing(target=voxel_spacing):
    """
    this function loads the CAT08 data and resamples it to voxel size 0.5*0.5*0.5 mmm
    the processed data will be stored in ./processed_training
    """

    dataset_dirs = ["../training/dataset0%d/image0%d.mhd" % (idx, idx) for idx in range(8)]

    for idx in range(8):
        print("\nprocessing dataset0" + str(idx))

        scan, _, spacing = load_itk(dataset_dirs[idx])
        spacing = spacing[::-1]
        print("spacing: " + str(spacing))

        # resample the images to achieve a voxel size of 0.5mm^3
        x_zoom, y_zoom, z_zoom = spacing / voxel_spacing
        print("initial shape: " + str(scan.shape))
        scan = ndimage.zoom(scan, (x_zoom, y_zoom, z_zoom))
        print("final shape: " + str(scan.shape))

        # save array as .npz file
        savez_compressed("../processed_training/%d.npz" % idx)

from preprocessing import *
from utils import load_points


if __name__ == "__main__":
    # change_voxel_spacing()
    dataset_idx, image_idx = 0, 0
    path = "./training/dataset0%d/vessel%s/reference.txt" % (dataset_idx, str(image_idx))
    points = load_points(path)
    reference_points = load_reference_points("./preprocessing/reference_directions.txt")

    print(points.shape, reference_points.shape)

    print(points[0, :3], points[20, :3])
    cat = find_direction_cat(points[0, :3], points[20, :3], reference_points)
    print(reference_points[cat])

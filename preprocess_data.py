from scipy import ndimage
from utils import load_itk
from numpy import savez_compressed
from warnings import filterwarnings
filterwarnings("ignore")

voxel_spacing = 0.5


if __name__ == "__main__":
    """
    this script loads the CAT08 data and resamples it to voxel size 0.5*0.5*0.5 mmm
    the processed data will be stored in ./processed_training
    """

    dataset_dirs = ["./training/dataset0%d/image0%d.mhd" % (idx, idx) for idx in range(8)]

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
        savez_compressed("./processed_training/%d.npz" % idx)

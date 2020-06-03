import sys
from preprocessing import *
import numpy as np
from utils.gpu import set_memory_growth
from model import create_model
from utils import load_points, segment_image, load_itk, world_to_voxel, create_fake_images
from os.path import join
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import math


assert len(sys.argv) == 4, "provide the epochs number, batch size, and batch per epoch"

training_dir = "./training"


def get_batch(dataset_idx, vessel_idx, batch_size=32):
    """
    :param dataset_idx: dataset to use
    :param vessel_idx: vessel to use
    :param batch_size: size of the batch gee
    :return: a batch of data from the given dataset and vessel
    """

    print("-", end="")
    reference_points = load_reference_points("./preprocessing/reference_directions.txt")
    probs, radii, directions, input_data = [], [], [], []

    points_path = join(training_dir, "dataset0%d/vessel%s/reference.txt" % (dataset_idx, str(vessel_idx)))
    points = load_points(points_path)

    image, _, _ = load_itk(join(training_dir, "dataset0%d/image0%d.mhd" % (dataset_idx, dataset_idx)))
    idxs = np.random.randint(300, len(points) - 300, batch_size)
    for idx in idxs:
        radius, direction = create_sample(idx, points, reference_points)

        point = world_to_voxel(points[idx, :3])
        patch = segment_image(image, point).copy()

        if patch.shape == (19, 19, 19):
            input_data.append(patch)
            probs.append(1.)
            radii.append(radius)
            directions.append(direction)

    input_data = np.asarray(input_data).reshape(-1, 19, 19, 19, 1)
    radii = np.asarray(radii).reshape(-1, 1)
    directions = np.asarray(directions).reshape(-1, 500)
    probs = np.asarray(probs).reshape(-1, 1)
    # print(input_data.shape, radii.shape, directions.shape, probs.shape)

    return input_data, [probs, radii, directions]


epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
batches_per_epoch = int(sys.argv[3])
total_iterations = epochs * batch_size * batches_per_epoch
print("epochs: %s\nbatch size: %d\nbatches per epoch: %d\ntotal iterations: %d" % (
    epochs, batch_size, batches_per_epoch, total_iterations))


def step_decay(epoch):
    initial_lrate = 1e-3
    epochs_drop = 10000 / batch_size / batches_per_epoch
    drop = 0.1
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate_callback = LearningRateScheduler(step_decay)
set_memory_growth()

"""
Each batch will be randomly selected from one vessel of one CCTA scan
Note that the direction generation method is not yet perfected
"""

if __name__ == "__main__":
    model = create_model()
    dataset_idx, image_idx = 0, 0
    for e in range(epochs):
        print("Epoch %d/%d\n[" % (e + 1, epochs), end="")
        for b in range(batches_per_epoch):
            # print("-", end="")
            X_batch, y_batch = get_batch(dataset_idx % 8, image_idx % 4, batch_size=batch_size)
            model.fit(X_batch, y_batch, verbose=0, epochs=1, callbacks=[lrate_callback])
            dataset_idx += 1
            image_idx += 1
        print("]")

    model.evaluate(X_batch, y_batch)
    model.save_weights("./models/model1.h5")

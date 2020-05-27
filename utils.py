from cnn import create_model
import tensorflow as tf
import numpy as np

W, D = 19, 500


def create_fake_images(value=0, N=10, w=W):
    return tf.constant(value, shape=(N, w, w, w, 1), dtype=tf.dtypes.float32)


def create_fake_labels(r=1, d=D, N=10, on=0):
    prob = np.zeros([N, d], dtype="float32")
    prob[:, 0] = 1
    return np.hstack([np.full(fill_value=on, shape=(N, 1)), np.full(fill_value=r, shape=(N, 1), dtype="float32"), prob])


if __name__ == "__main__":
    model = create_model(W, D)
    preds = model.predict(create_fake_images())
    # print(preds[0].shape, preds[1].shape)
    model.evaluate(create_fake_images(N=5), create_fake_labels(N=5), verbose=1)
    # print(len(create_fake_labels()))

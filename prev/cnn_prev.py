from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, Flatten, Concatenate
from tensorflow.keras import Model, backend
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

IMAGE_FORMAT = backend.image_data_format()


def conv_block(input_, n_filters=32, kernel_size=(3,3,3), dilation_rate=1, idx=0):
    """
    create a conv_block with the following structure:
    Conv3D -- BatchNormalization -- ReLU

    :param input_: input to the conv_block
    :param n_filters: number of filters in Conv3D
    :param kernel_size: kernel size of Conv3D
    :param dilation_rate: dilation rate of Conv3D
    :param idx: index of conv_block for naming
    :return: x (output of relu)
    """

    idx = str(idx)
    x = Conv3D(n_filters, kernel_size, dilation_rate=dilation_rate, name="conv" + idx)(input_)
    x = BatchNormalization(name="bn" + idx)(x)
    x = Activation("relu", name="relu" + idx)(x)

    return x


def create_model(w=19, D=500, initial_lr=0.001):
    """
    create a CNN for coronary artery centerline extraction

    :param initial_lr: initial learning rate for Adam optimizer
    :param w: on of the three input dimensions of the isotropic 3D image
    :param D: number of categories of directions
    :return: constructed model
    """

    if IMAGE_FORMAT == "channels_last":
        inputs = Input(shape=(w, w, w, 1), name="input")
    else:
        inputs = Input(shape=(1, w, w, w), name="input")

    x = conv_block(inputs, 32, (3, 3, 3), 1, idx=1)
    x = conv_block(x, 32, (3, 3, 3), 1, idx=2)
    x = conv_block(x, 32, (3, 3, 3), 2, idx=3)
    x = conv_block(x, 32, (3, 3, 3), 4, idx=4)

    # tracker
    x = conv_block(x, 64, (3, 3, 3), 1, idx=5)
    x = conv_block(x, 64, (1, 1, 1), 1, idx=6)
    x = Conv3D(D+1, (1, 1, 1), dilation_rate=1, name="conv7")(x)
    x = Flatten(name="flatten")(x)
    x = final_activation(x)
    x = Concatenate(name="concatenate")(x)

    # # discriminator
    # x = conv_block(x, 64, (3, 3, 3), 1, idx=5)
    # x = conv_block(x, 64, (1, 1, 1), 1, idx=6)
    # x = Conv3D(D + 1, (1, 1, 1), dilation_rate=1, name="conv7")(x)
    # x = Flatten(name="flatten")(x)

    model = Model(inputs=inputs, outputs=x)
    schedule = PiecewiseConstantDecay([i * 10000 for i in range(1, 6)], [initial_lr * (0.1 ** i) for i in range(0, 6)])
    optimizer = Adam(learning_rate=schedule)
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model


def final_activation(input_):
    """
    final activation of the model
    the first output has a linear activation (no activation) for radius estimation
    the second output has a softmax activation for estimating the probability distribution over the D directions

    :param input_: input to the final activations
    :return: the two outputs [radius, directions]
    """

    output1 = input_[..., :1]
    output2 = Activation("softmax", name="direction_output")(input_[..., 1:])
    return [output1, output2]


def custom_loss(y_true, y_pred, lambda_r=10):
    """
    custom loss function that is a combination of mean squared error and categorical cross entropy loss
    """

    reg_loss = lambda_r * mean_squared_error(y_true[:, :1], y_pred[:, :1])
    clf_loss = categorical_crossentropy(y_true[:, 1:], y_pred[:, 1:])
    return reg_loss + clf_loss


if __name__ == "__main__":
    model = create_model()
    model.summary()

from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, Flatten, Concatenate
from tensorflow.keras import Model, backend
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

IMAGE_FORMAT = backend.image_data_format()


def conv_block(input_, n_filters=32, kernel_size=(3, 3, 3), dilation_rate=1, idx="O"):
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

    x = conv_block(inputs, 32, (3, 3, 3), 1, idx="1")
    x = conv_block(x, 32, (3, 3, 3), 1, idx="2")
    x = conv_block(x, 32, (3, 3, 3), 2, idx="3")
    x = conv_block(x, 32, (3, 3, 3), 4, idx="4")

    # tracker
    x_t = conv_block(x, 64, (3, 3, 3), 1, idx="5_t")
    x_t = conv_block(x_t, 64, (1, 1, 1), 1, idx="6_t")
    x_t = Conv3D(D+1, (1, 1, 1), dilation_rate=1, name="conv7_t")(x_t)
    x_t = Flatten(name="flatten_t")(x_t)
    x_t = final_activation(x_t)

    # discriminator
    x_d = conv_block(x, 64, (3, 3, 3), 1, idx="5_d")
    x_d = conv_block(x_d, 64, (1, 1, 1), 1, idx="6_d")
    x_d = Conv3D(1, (1, 1, 1), dilation_rate=1, name="conv7_d", activation="sigmoid")(x_d)
    x_d = Flatten(name="flatten_d")(x_d)

    outputs = Concatenate()([x_d, x_t[0], x_t[1]])

    model = Model(inputs=inputs, outputs=outputs)
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


def custom_loss(y_true, y_pred, lambda_r=15, beta=1):
    """
    custom loss function that is a combination of mean squared error and categorical cross entropy loss
    """

    disc_loss = beta * binary_crossentropy(y_true[:, :1], y_pred[:, :1])
    reg_loss = lambda_r * mean_squared_error(y_true[:, 1:2], y_pred[:, 1:2])
    clf_loss = categorical_crossentropy(y_true[:, 2:], y_pred[:, 2:])
    return disc_loss + reg_loss + clf_loss


if __name__ == "__main__":
    model = create_model()
    model.summary()

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow import keras

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs):
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]


class MaxUnpooling2D(Layer):
    def __init__(self, **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * 2,
                    input_shape[2] * 2,
                    input_shape[3],
                )
            self.output_shape1 = output_shape
            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        shape = input_shape[1]
        return (shape[0], shape[1] * 2, shape[2] * 2, shape[3])


class MaxPoolingWithArgmax2DA(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2DA, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding,
            }
        )
        return config

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, *pool_size, 1]
        padding = padding.upper()
        strides = [1, *strides, 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=strides, padding=padding
        )

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2DA(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2DA, self).__init__(**kwargs)
        self.size = size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
            }
        )
        return config

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = K.cast(mask, "int32")
        input_shape = tf.shape(updates, out_type="int32")

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3],
            )

        ret = tf.scatter_nd(
            K.expand_dims(K.flatten(mask)), K.flatten(updates), [K.prod(output_shape)]
        )

        input_shape = updates.shape
        out_shape = [
            -1,
            input_shape[1] * self.size[0],
            input_shape[2] * self.size[1],
            input_shape[3],
        ]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )


class CAM(Layer):
    def __init__(self, previous_layer, shape, mode="thr", thr=0, **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.previous_layer = previous_layer
        self.cam_weights = tf.reshape(previous_layer.get_weights()[0], [-1])
        self.input_size = shape
        self.mode = mode
        self.thr = thr

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "previous_layer": self.previous_layer,
                "shape": self.input_size,
                "mode": self.mode,
                "thr": self.thr,
            }
        )

        return config

    def get_weights(self):
        return [self.cam_weights]

    def call(self, inputs, output_shape=None):
        conv_output = inputs[0]
        output = inputs[1]

        if output_shape is None:
            output_shape = [self.input_size, conv_output]

        cam = tf.reduce_sum(conv_output * self.cam_weights, axis=3)

        return cam, output

    def compute_output_shape(self, input_shape):
        return [self.input_size, input_shape]


def simple_model(shape):
    """define the most simple fcnn model

    Args:
        shape (tuple(int,int,int)): the shape of the input data
    """

    input = keras.layers.Input(shape=shape)
    conv1 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(input)
    maxpool1 = keras.layers.MaxPool2D((2, 2))(conv1)
    upsampling1 = keras.layers.UpSampling2D((2, 2))(maxpool1)
    conv2 = keras.layers.Conv2D(3, (3, 3), padding="same", activation="softmax")(
        upsampling1
    )

    model = keras.Model(inputs=input, outputs=conv2)

    return model


def simple_model_multi(shape):
    """define the most simple fcnn model

    Args:
        shape (tuple(Int,Int,Int)): the shape of the input data
    """

    input = keras.layers.Input(shape=shape)
    conv1 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(input)
    maxpool1 = keras.layers.MaxPool2D((2, 2))(conv1)
    upsampling1 = keras.layers.UpSampling2D((2, 2))(maxpool1)
    conv2 = keras.layers.Conv2D(3, (3, 3), padding="same", activation="softmax")(
        upsampling1
    )

    model = keras.Model(inputs=input, outputs=conv2)

    return model


def simple_model_classification(shape):
    """define the most simple cnn classification model

    Args:
        shape (tuple(Int,Int,Int)): the shape of the input data
    """

    inputs = keras.layers.Input(shape=shape)
    conv1 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(inputs)
    conv2 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(conv1)
    maxpool1 = keras.layers.MaxPool2D((2, 2))(conv2)
    conv3 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(maxpool1)
    conv4 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(conv3)
    up1 = keras.layers.UpSampling2D((2, 2))(conv4)
    gap1 = keras.layers.GlobalAveragePooling2D()(up1)
    dense1 = keras.layers.Dense(1, activation="sigmoid")(gap1)

    model = keras.Model(inputs=inputs, outputs=dense1)

    return model


def brain_classification(shape):
    """define the most simple cnn classification model

    Args:
        shape (tuple(Int,Int,Int)): the shape of the input data
    """

    inputs = keras.layers.Input(shape=shape)

    conv1 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
    conv2 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(conv1)
    conv3 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(conv2)
    maxpool1 = keras.layers.MaxPool2D((2, 2))(conv3)

    conv4 = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(maxpool1)
    conv5 = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(conv4)
    conv6 = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(conv5)
    maxpool2 = keras.layers.MaxPool2D((2, 2))(conv6)

    up1 = keras.layers.UpSampling2D((4, 4))(maxpool2)
    gap1 = keras.layers.GlobalAveragePooling2D()(up1)

    dense1 = keras.layers.Dense(1, activation="sigmoid")(gap1)

    model = keras.Model(inputs=inputs, outputs=dense1)

    return model


def brain_simple_classification(shape):
    """define the most simple cnn classification model

    Args:
        shape (tuple(Int,Int,Int)): the shape of the input data
    """

    inputs = keras.layers.Input(shape=shape)

    conv1 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
    conv2 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(conv1)
    conv3 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(conv2)
    maxpool1 = keras.layers.MaxPool2D((2, 2))(conv3)

    up1 = keras.layers.UpSampling2D((2, 2))(maxpool1)
    gap1 = keras.layers.GlobalAveragePooling2D()(up1)

    dense1 = keras.layers.Dense(1, activation="sigmoid")(gap1)

    model = keras.Model(inputs=inputs, outputs=dense1)

    return model


def concatenate_simple(shape):
    inputs = keras.layers.Input(shape=shape)
    x1 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(inputs)
    x2 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(inputs)
    x3 = keras.layers.Concatenate(axis=3)([x1, x2])
    x4 = keras.layers.MaxPool2D((2, 2))(x3)
    x5 = keras.layers.UpSampling2D((2, 2))(x4)
    output = keras.layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x5)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


def fcn_vgg16(shape, version=8):
    input = keras.layers.Input(shape=shape)

    # first layer
    conv1 = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(input)
    conv2 = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv1)
    mp1 = keras.layers.MaxPool2D((2, 2))(conv2)

    # second layer
    conv3 = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(mp1)
    conv4 = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)
    mp2 = keras.layers.MaxPool2D((2, 2))(conv4)

    # third layer
    conv5 = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(mp2)
    conv6 = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(conv5)
    conv7 = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(conv6)
    mp3 = keras.layers.MaxPool2D((2, 2))(conv7)

    # fourth layer
    conv8 = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(mp3)
    conv9 = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(conv8)
    conv10 = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(conv9)
    mp4 = keras.layers.MaxPool2D((2, 2))(conv10)

    # fifth layer
    conv11 = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(mp4)
    conv12 = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(conv11)
    conv13 = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(conv12)
    mp5 = keras.layers.MaxPool2D((2, 2))(conv13)

    # last layer
    conv14 = keras.layers.Conv2D(1, (1, 1), padding="same")(mp5)
    # replarce sigmoid layer
    output32 = tf.math.sigmoid(conv14)
    output32 = keras.layers.UpSampling2D((32, 32))(output32)

    _mp4 = keras.layers.Conv2D(1, (1, 1), padding="same")(mp4)
    us2 = keras.layers.UpSampling2D((2, 2))(conv14)
    add16 = keras.layers.Add()([us2, _mp4])
    # replarce sigmoid layer
    output16 = tf.math.sigmoid(add16)
    output16 = keras.layers.UpSampling2D((16, 16))

    _mp3 = keras.layers.Conv2D(1, (1, 1), padding="same")(mp3)
    us3 = keras.layers.UpSampling2D((2, 2))(add16)
    add8 = keras.layers.Add()([us3, mp3])
    # replarce sigmoid layer
    output8 = tf.math.sigmoid(add8)
    output8 = keras.layers.UpSampling2D((8, 8))(output8)

    if version == 32:
        return keras.Model(inputs=input, outputs=output32)
    elif version == 16:
        return keras.Model(inputs=input, outputs=output16)
    elif version == 8:
        return keras.Model(inputs=input, outputs=output8)
    else:
        print("please confirme versions arguments")
        assert False


def test_model(shape):
    """define the most simple fcnn model

    Args:
        shape (tuple(Int,Int,Int)): the shape of the input data
    """

    print(shape)
    input = keras.layers.Input(shape=shape)
    conv1 = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(input)
    maxpool1 = keras.layers.MaxPool2D((2, 2))(conv1)
    upsampling1 = keras.layers.UpSampling2D((2, 2))(maxpool1)
    conv2 = keras.layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")(
        upsampling1
    )

    model = keras.Model(inputs=input, outputs=conv2)

    return model


def U_Net(shape):
    input = keras.layers.Input(shape=shape)

    # Encoder
    # block1
    b1c1 = tf.keras.layers.Conv2D(
        64, (3, 3), name="block1_conv1", activation="relu", padding="same"
    )(input)
    b1c2 = tf.keras.layers.Conv2D(64, (3, 3), name="block1_conv2", padding="same")(b1c1)
    b1bn1 = tf.keras.layers.BatchNormalization()(b1c2)
    b1act = tf.keras.layers.ReLU()(b1bn1)
    b1p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp1")(b1act)

    # block2
    b2c1 = tf.keras.layers.Conv2D(
        128, (3, 3), name="block2_conv1", activation="relu", padding="same"
    )(b1p)
    b2c2 = tf.keras.layers.Conv2D(128, (3, 3), name="block2_conv2", padding="same")(
        b2c1
    )
    b2bn1 = tf.keras.layers.BatchNormalization()(b2c2)
    b2act = tf.keras.layers.ReLU()(b2bn1)
    b2p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp2")(b2act)

    # block3
    b3c1 = tf.keras.layers.Conv2D(
        256, (3, 3), name="block3_conv1", activation="relu", padding="same"
    )(b2p)
    b3c2 = tf.keras.layers.Conv2D(256, (3, 3), name="block3_conv2", padding="same")(
        b3c1
    )
    b3bn1 = tf.keras.layers.BatchNormalization()(b3c2)
    b3act = tf.keras.layers.ReLU()(b3bn1)
    b3p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp3")(b3act)

    # block4
    b4c1 = tf.keras.layers.Conv2D(
        512, (3, 3), name="block4_conv1", activation="relu", padding="same"
    )(b3p)
    b4c2 = tf.keras.layers.Conv2D(512, (3, 3), name="block4_conv2", padding="same")(
        b4c1
    )
    b4bn1 = tf.keras.layers.BatchNormalization()(b4c2)
    b4act = tf.keras.layers.ReLU()(b4bn1)
    b4p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp4")(b4act)

    # block5
    b5c1 = tf.keras.layers.Conv2D(
        1024, (3, 3), name="block5_conv1", activation="relu", padding="same"
    )(b4p)
    b5c2 = tf.keras.layers.Conv2D(1024, (3, 3), name="block5_conv2", padding="same")(
        b5c1
    )
    b5bn1 = tf.keras.layers.BatchNormalization()(b5c2)
    b5act = tf.keras.layers.ReLU()(b5bn1)

    # Decoder
    # block6
    b6up = tf.keras.layers.UpSampling2D((2, 2))(b5act)
    b6c1 = tf.keras.layers.Conv2D(
        512, (2, 2), name="block6_conv1", activation="relu", padding="same"
    )(b6up)
    b7conc = tf.keras.layers.Concatenate(axis=3)([b4act, b6c1])
    b6c2 = tf.keras.layers.Conv2D(
        512, (3, 3), name="block6_conv2", activation="relu", padding="same"
    )(b6c1)
    b6c3 = tf.keras.layers.Conv2D(
        512, (3, 3), name="block6_conv3", activation="relu", padding="same"
    )(b6c2)
    b6bn = tf.keras.layers.BatchNormalization()(b6c3)
    b6act = tf.keras.layers.ReLU()(b6bn)

    # block7
    b7up = tf.keras.layers.UpSampling2D((2, 2))(b6act)
    b7c1 = tf.keras.layers.Conv2D(
        256, (2, 2), name="block7_conv1", activation="relu", padding="same"
    )(b7up)
    b7conc = tf.keras.layers.Concatenate(axis=3)([b3act, b7c1])
    b7c2 = tf.keras.layers.Conv2D(
        256, (3, 3), name="block7_conv2", activation="relu", padding="same"
    )(b7conc)
    b7c3 = tf.keras.layers.Conv2D(
        256, (3, 3), name="bloc7_conv3", activation="relu", padding="same"
    )(b7c2)
    b7bn = tf.keras.layers.BatchNormalization()(b7c3)
    b7act = tf.keras.layers.ReLU()(b7bn)

    # block8
    b8up = tf.keras.layers.UpSampling2D((2, 2))(b7act)
    b8c1 = tf.keras.layers.Conv2D(
        128, (2, 2), name="block8_conv1", activation="relu", padding="same"
    )(b8up)
    b8conc = tf.keras.layers.Concatenate(axis=3)([b2act, b8c1])
    b8c2 = tf.keras.layers.Conv2D(
        128, (3, 3), name="block8_conv2", activation="relu", padding="same"
    )(b8conc)
    b8c3 = tf.keras.layers.Conv2D(
        128, (3, 3), name="block8_conv3", activation="relu", padding="same"
    )(b8c2)
    b8bn = tf.keras.layers.BatchNormalization()(b8c3)
    b8relu = tf.keras.layers.ReLU()(b8bn)

    # block9
    b9up = tf.keras.layers.UpSampling2D((2, 2))(b8relu)
    b9c1 = tf.keras.layers.Conv2D(
        64, (2, 2), name="block9_conv1", activation="relu", padding="same"
    )(b9up)
    b9conc = tf.keras.layers.Concatenate(axis=3)([b1act, b9c1])
    b9c2 = tf.keras.layers.Conv2D(
        64, (3, 3), name="block9_conv2", activation="relu", padding="same"
    )(b9conc)
    b9c3 = tf.keras.layers.Conv2D(
        64, (3, 3), name="block9_conv3", activation="relu", padding="same"
    )(b9c2)
    b9bn = tf.keras.layers.BatchNormalization()(b9c3)
    b9relu = tf.keras.layers.ReLU()(b9bn)

    # block10
    b10up = tf.keras.layers.UpSampling2D((2, 2))(b9relu)
    output = tf.keras.layers.Conv2D(1, (2, 2), activation="sigmoid", padding="same")(
        b10up
    )

    return keras.Model(inputs=input, outputs=output)


def mini_U_Net(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)

    # Encoder
    # block1
    b1c1 = tf.keras.layers.Conv2D(
        16, (2, 2), name="block1_conv1", activation="relu", padding="same"
    )(input_layer)
    b1p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp1")(b1c1)

    # block2
    b2c1 = tf.keras.layers.Conv2D(
        32, (2, 2), name="block2_conv1", activation="relu", padding="same"
    )(b1p)
    b2p = tf.keras.layers.MaxPool2D((2, 2), name="block2_mp1")(b2c1)

    # block3
    b3c1 = tf.keras.layers.Conv2D(
        64, (2, 2), name="block3_conv1", activation="relu", padding="same"
    )(b2p)
    b3p = tf.keras.layers.MaxPool2D((2, 2), name="block3_mp1")(b3c1)

    # block4
    b4c1 = tf.keras.layers.Conv2D(
        128, (2, 2), name="block4_conv1", activation="relu", padding="same"
    )(b3p)
    b4p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp4")(b4c1)

    # block5
    b5c1 = tf.keras.layers.Conv2D(
        256, (2, 2), name="block5_conv1", activation="relu", padding="same"
    )(b4p)

    # Decoder
    # block6
    b6up = tf.keras.layers.UpSampling2D((2, 2))(b5c1)
    # b6conc = tf.keras.layers.Concatenate(axis=3)([b4c1,b6up])
    b6c1 = tf.keras.layers.Conv2D(
        128, (3, 3), name="block6_conv1", activation="relu", padding="same"
    )(b6up)

    # block7
    b7up = tf.keras.layers.UpSampling2D((2, 2))(b6c1)
    b7conc = tf.keras.layers.Concatenate(axis=3)([b3c1, b7up])
    b7c1 = tf.keras.layers.Conv2D(
        64, (3, 3), name="block7_conv1", activation="relu", padding="same"
    )(b7conc)

    # block8
    b8up = tf.keras.layers.UpSampling2D((2, 2))(b7c1)
    # b8conc = tf.keras.layers.Concatenate(axis=3)([b2c1,b8up])
    b8c1 = tf.keras.layers.Conv2D(
        32, (3, 3), name="block8_conv1", activation="relu", padding="same"
    )(b8up)

    # block9
    b9up = tf.keras.layers.UpSampling2D((2, 2))(b8c1)
    b9conc = tf.keras.layers.Concatenate(axis=3)([b1c1, b9up])
    b9c1 = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(b9conc)

    return keras.Model(inputs=input_layer, outputs=b9c1)


def mini_U_Net_v2(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)

    # Encoder
    # block1
    b1c1 = tf.keras.layers.Conv2D(
        32, (2, 2), name="block1_conv1", activation="relu", padding="same"
    )(input_layer)
    b1p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp1")(b1c1)

    # block2
    b2c1 = tf.keras.layers.Conv2D(
        64, (2, 2), name="block2_conv1", activation="relu", padding="same"
    )(b1p)
    b2p = tf.keras.layers.MaxPool2D((2, 2), name="block2_mp1")(b2c1)

    # block3
    b3c1 = tf.keras.layers.Conv2D(
        128, (2, 2), name="block3_conv1", activation="relu", padding="same"
    )(b2p)
    b3p = tf.keras.layers.MaxPool2D((2, 2), name="block3_mp1")(b3c1)

    # block4
    b4c1 = tf.keras.layers.Conv2D(
        256, (2, 2), name="block4_conv1", activation="relu", padding="same"
    )(b3p)
    b4p = tf.keras.layers.MaxPool2D((2, 2), name="block1_mp4")(b4c1)

    # Decoder
    # block5
    b5c1 = tf.keras.layers.Conv2D(
        512, (2, 2), name="block5_conv1", activation="relu", padding="same"
    )(b4p)

    # block6
    b6ct = tf.keras.layers.Conv2DTranspose(
        256, (2, 2), strides=(2, 2), name="block6_ct", activation="relu", padding="same"
    )(b5c1)
    b6c1 = tf.keras.layers.Conv2D(
        256, (2, 2), name="blcok6_c1", activation="relu", padding="same"
    )(b6ct)

    # block7
    b7ct = tf.keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), name="block7_ct", activation="relu", padding="same"
    )(b6c1)
    b7conc = tf.keras.layers.Concatenate(axis=3)([b3c1, b7ct])
    b7c1 = tf.keras.layers.Conv2D(
        128, (3, 3), name="block7_conv1", activation="relu", padding="same"
    )(b7conc)

    # block8
    b8ct = tf.keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), name="block8_ct", activation="relu", padding="same"
    )(b7c1)
    b8conc = tf.keras.layers.Concatenate(axis=3)([b8ct, b2c1])
    b8c1 = tf.keras.layers.Conv2D(
        64, (3, 3), name="block8_conv1", activation="relu", padding="same"
    )(b8conc)

    # block9
    b9ct = tf.keras.layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), name="block9_ct", activation="relu", padding="same"
    )(b8c1)
    b9conc = tf.keras.layers.Concatenate(axis=3)([b9ct, b1c1])
    b9c1 = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(b9conc)

    return keras.Model(inputs=input_layer, outputs=b9c1)


def u_net_encoder_block(input, channel):
    conv = tf.keras.layers.Conv2D(channel, (2, 2), activation="relu", padding="same")(
        input
    )
    output = tf.keras.layers.MaxPool2D((2, 2))(conv)

    return conv, output, channel * 2


def u_net_decoder_block(input, skip_input, channel, last=False):
    up = tf.keras.layers.UpSampling2D((2, 2))(input)
    if last:
        channel = 1
        conc = tf.keras.layers.Concatenate(axis=3)([up, skip_input])
        conv = tf.keras.layers.Conv2D(
            channel, (2, 2), activation="sigmoid", padding="same"
        )(conc)
    else:
        channel /= 2
        conc = tf.keras.layers.Concatenate(axis=3)([up, skip_input])
        conv = tf.keras.layers.Conv2D(
            channel, (2, 2), activation="relu", padding="same"
        )(conc)

    return conv, channel


def mini_u_net(channel, input_shape, depth):
    input_layer = keras.layers.Input(shape=input_shape)
    output = input_layer

    middle_output = []
    for i in range(depth):
        conv, output, channel = u_net_encoder_block(output, channel)
        middle_output.insert(0, conv)

    output = tf.keras.layers.Conv2D(channel, (2, 2), activation="relu", padding="same")(
        output
    )

    for j in range(depth):
        if depth - 1 == j:
            output, channel = u_net_decoder_block(
                output, middle_output[j], channel, last=True
            )
        else:
            output, channel = u_net_decoder_block(output, middle_output[j], channel)

    return keras.Model(inputs=input_layer, outputs=output)


class MaxPool2DWithArgmax(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxPool2DWithArgmax, self).__init__()

    def call(self, input):
        pool, pool_index = tf.nn.max_pool_with_argmax(
            input, (2, 2), strides=2, padding="SAME"
        )
        return [pool, pool_index]


class UnPooling2D(tf.keras.layers.Layer):
    def __init__(self):
        super(UnPooling2D, self).__init__()

    def call(self, input_index):
        input, index = input_index[0], input_index[1]
        input_shape = tf.shape(input, out_type=tf.int64)
        output_shape = [
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            input_shape[3],
        ]
        input_vector = tf.reshape(input, [-1])
        pool_index_vector = tf.reshape(index, [-1, 1])
        unpool = tf.scatter_nd(
            pool_index_vector,
            input_vector,
            [output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]],
        )
        output = tf.reshape(unpool, output_shape)
        return output


def seg_net_encoder(input, channel):
    conv = tf.keras.layers.Conv2D(channel, (2, 2), activation="relu", padding="same")(
        input
    )
    pool, pool_index = MaxPoolingWithArgmax2D()(conv)

    return pool, pool_index, channel * 2


def seg_net_decoder(input, pool_index, channel):
    channel /= 2
    unpool = MaxUnpooling2D()([input, pool_index])
    conv = tf.keras.layers.Conv2D(channel, (2, 2), activation="relu", padding="same")(
        unpool
    )

    return conv, channel


def seg_net(channel, input_shape, depth):
    input_layer = keras.layers.Input(shape=input_shape)
    output = input_layer

    pooling_indexes = []
    for i in range(depth):
        output, pooling_index, channel = seg_net_encoder(output, channel)
        pooling_indexes.insert(0, pooling_index)

    channel /= 2

    for i in range(depth):
        output, channel = seg_net_decoder(output, pooling_indexes[i], channel)

    output_layer = tf.keras.layers.Conv2D(
        1, (2, 2), activation="sigmoid", padding="same"
    )(output)

    return keras.Model(inputs=input_layer, outputs=output_layer)


def model_plot(model):
    plot_model(model, show_shapes=True)


def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def IU(y_true, y_pred):
    """this functin caliculate IoU metrics between y_true and y_pred
    Assuming that y_true and y_pred is consisted of 0 or 1 and their shape is (B,H,W,C)
    """
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (
        tf.reduce_sum(y_true, [1, 2, 3])
        + tf.reduce_sum(y_pred, [1, 2, 3])
        - intersection
    )

    return tf.math.reduce_mean(intersection / union, axis=0)

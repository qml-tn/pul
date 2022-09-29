import numpy as np
import tensorflow as tf
from tnpul.tfmps.utils import basis_f


def spiral_inds(row):
    lis = [[0 for i in range(0, row)] for j in range(0, row)]
    s = []
    if row > 1:
        s += [row - 1]
        for i in range(row - 1, 0, -1):
            s += [i, i]
    b = 1
    e = 0
    a = 0
    c = 0
    d = 0
    lis[0][0] = e
    for n in s:
        for f in range(n):
            c += a
            d += b
            e += 1
            lis[c][d] = e
        a, b = b, -a
    return list(np.reshape(lis, [-1]))


class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, type="angle", d=2, flatten_off=False, *args, **kwargs):
        super(FeatureEmbedding, self).__init__(*args, **kwargs)
        self.type = type
        self.flatten = None
        if not flatten_off:
            self.flatten = tf.keras.layers.Flatten()
        self.d = d
        if d != 2:
            raise Exception(
                f'Feature dimension {d}. Allowed feature dimension is 2.')
        self.config = {
            'type': type,
            'd': d,
            'flatten_off': flatten_off,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def call(self, input):
        pihalf = np.pi/2.0
        if self.flatten is not None:
            x = self.flatten(input)
        x = input
        x = tf.transpose(x)
        if self.type == "angle":
            x = tf.stack(
                [tf.math.cos(x*pihalf), tf.math.sin(x*pihalf)], axis=-1)
        elif self.type == "linear":
            x = tf.stack([1-x, x], axis=-1)
        else:
            x = tf.stack(
                [tf.math.cos(x*pihalf), tf.math.sin(x*pihalf)], axis=-1)
        return x


class DecodeFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, type="angle", name="decoder", *args, **kwargs):
        super(DecodeFeatureEmbedding, self).__init__(
            name=name, *args, **kwargs)
        self.type = type
        self.config = {
            'type': type,
            'name': name
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def call(self, input):
        # input.shape (N,nbatch,d)
        pihalf = np.pi/2.0
        x = input[:, :, 0]
        y = input[:, :, 1]
        x = tf.math.atan2(y, x)
        x = tf.transpose(x)
        # x.shape  (nbatch,N)
        return x/pihalf


class SimpleMPSEmbedding(tf.keras.layers.Layer):
    """ Layer that converts the image input into an MPS
    d:          local Hilbert space dimension
    embedding:  Embedding type: "angle" (default), "linear" 
    mixing:     Determines if mixing is used if we have a batch input
    aug_phi:    Random shift of the encoding  x += aug_phi 0.001 (default) .
    """

    def __init__(self, d=2, permute=False, spiral=False, embedding="angle", aug_phi=1e-3, name="simple_mps_embedding"):
        super(SimpleMPSEmbedding, self).__init__(name=name)
        self.d = d
        self.permute = permute
        self.spiral = spiral
        self.embedding = embedding
        self.aug_phi = aug_phi

        if spiral and permute:
            logging.warning(
                "Warning: permute and spiral can not be both set to True")
            logging.warning("Using permute=False")
            self.permute = False
        self.config = {
            'd': d,
            'permute': permute,
            'spiral': spiral,
            'embedding': embedding,
            'aug_phi': aug_phi,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def build(self, input_shape):
        '''
          The first dimension is the batch_size.
          The second dimension should be the number of channes: channels_first format
        '''
        self.n = np.prod(input_shape[1:])

        self.flatten = tf.keras.layers.Flatten()

        if self.permute:
            logging.info("Using a random permutation of the input values")
            self.inds = np.random.permutation(self.n)
        elif self.spiral:
            # Here we assume that the image is a square image and n=row**2
            row = int(np.sqrt(self.n))
            self.inds = spiral_inds(row)
        else:
            self.inds = range(self.n)

    # Helper function that encodes a batch of numbers into local hilbert space
    # this matrix will be used as a part of a larger MPS
    def to_mps_matrix(self, x):
        if self.embedding == "linear":
            x = tf.stack(
                [1 - x, x], axis=1)
        else:
            pi_half = np.pi/2.
            cx = tf.math.cos(x*pi_half)
            sx = tf.math.sin(x*pi_half)
            x = tf.stack([cx, sx], axis=1)

        return tf.reshape(x, [-1, 1, 2, 1])

    def call(self, input, training=False):
        x = self.flatten(input)
        if training and x.shape[0] is not None:
            x += tf.random.normal(x.shape,
                                  mean=0.0, stddev=self.aug_phi)

        x = tf.clip_by_value(x, 0., 1.)

        n = x.shape[1]
        bs = x.shape[0]
        if n != self.n:
            raise Exception(
                f"Length of the input {n} and mps {self.n} indices do not agree.")
        mps = []
        for i in self.inds:
            A = self.to_mps_matrix(x[:, i])
            mps.append(A)
        return mps


class ImageMPSEmbedding(tf.keras.layers.Layer):
    """ Layer that converts the image input into an MPS
    d:          local Hilbert space dimension
    embedding:  Embedding type: "angle" (default), "linear" 
    mixing:     Determines if mixing is used if we have a batch input
    aug_phi:    Random shift of the encoding  x += aug_phi 0.001 (default) .
    """

    def __init__(self, d=2, permute=False, spiral=False, embedding="angle", aug_phi=1e-3, name="image_mps_embedding"):
        super(ImageMPSEmbedding, self).__init__(name=name)
        self.d = d
        self.permute = permute
        self.spiral = spiral
        self.embedding = embedding
        self.aug_phi = aug_phi

        self.config = {
            "d": d,
            "permute": permute,
            "spiral": spiral,
            "embedding": embedding,
            "aug_phi": aug_phi,
            "name": name
        }

        if spiral and permute:
            logging.warning(
                "Warning: permute and spiral can not be both set to True")
            logging.warning("Using permute=False")
            self.permute = False

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def build(self, input_shape):
        '''
          The first dimension is the batch_size.
          The second dimension should be the number of channes: channels_first format
        '''
        self.n = np.prod(input_shape[2:])
        self.nchan = input_shape[1]
        self.d = self.nchan

        assert self.nchan in [
            1, 3], f"Number of channels not should be 1 or 3, but is {self.nchan}"

        self.reshape = tf.keras.layers.Reshape(target_shape=(self.nchan, -1))
        self.multiply = tf.keras.layers.Multiply()

        if self.permute:
            logging.info("Using a random permutation of the input values")
            self.inds = np.random.permutation(self.n)
        elif self.spiral:
            # Here we assume that the image is a square image and n=row**2
            row = int(np.sqrt(self.n))
            self.inds = spiral_inds(row)
        else:
            self.inds = range(self.n)

    # Helper function that encodes a batch of numbers into local hilbert space
    # this matrix will be used as a part of a larger MPS
    def to_mps_matrix(self, x, training=False):
        if training and x.shape[0] is not None:
            x += tf.random.normal(x.shape,
                                  mean=0.0, stddev=self.aug_phi)
        x = x/self.nchan
        x = tf.clip_by_value(x, 0., 1.)

        if self.embedding == "linear":
            x = tf.concat(
                [1 - tf.reduce_sum(x, axis=1, keepdims=True), x], axis=1)
        else:
            pi_half = np.pi/2.
            cx = tf.math.cos(x*pi_half)
            sx = tf.math.sin(x*pi_half)
            xlist = [tf.reduce_prod(cx, axis=1, keepdims=True)]
            for i in range(1, self.nchan):
                x1 = tf.reduce_prod(cx[i:], axis=1, keepdims=True)
                x2 = tf.reduce_prod(sx[:i], axis=1, keepdims=True)
                xlist.append(self.multiply([x1, x2]))
            xlist.append(tf.reduce_prod(sx, axis=1, keepdims=True))
            x = tf.concat(xlist, axis=1)

        return tf.reshape(x, [-1, 1, self.nchan+1, 1])

    def call(self, input):
        x = self.reshape(input)
        n = x.shape[2]
        if n != self.n:
            raise Exception(
                f"Length of the input {n} and mps {self.n} indices do not agree.")
        nchan = x.shape[1]
        if nchan != self.nchan:
            raise Exception(
                f"Number of channels {nchan} and mps {self.nchan} channels do not agree.")
        mps = []
        for i in self.inds:
            A = self.to_mps_matrix(x[:, :, i])
            mps.append(A)
        return mps


class FourierEmbedding(tf.keras.layers.Layer):
    def __init__(self, d=2, repeat=1, basis=["sin"]):
        super(FourierEmbedding, self).__init__()
        self.d = d
        assert len(
            basis) >= repeat, "There should be as many basis choices as repetitions"
        self.repeat = repeat
        self.flatten = tf.keras.layers.Flatten()
        self.basis = basis
        self.config = {
          "d": d,
          "repeat": repeat,
          "basis": basis
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def call(self, input):
        d = self.d
        x = self.flatten(input)  # (nbatch, N)
        x = tf.transpose(x)  # (N, nbatch)
        out = []
        for i in range(self.repeat):
            pi_half = np.pi/2
            emb = []
            for j in range(self.d):
                emb.append(basis_f(x, j, basis=self.basis[i]))
            out.append(tf.stack(emb, axis=-1) /
                       tf.math.sqrt(1.0*d))  # (N, nbatch, d)
        return tf.concat(out, axis=0)  # (N*repeat, nbatch, d)

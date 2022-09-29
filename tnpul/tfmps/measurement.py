import numpy as np
import tensorflow as tf
from tnpul.utils.mps import bond_dimension


class MPSMeasurement(tf.keras.layers.Layer):
    """MPS layer that is used for classification
    n:          number of classical parameters
    nout:       number of classes
    D:          Maximum bond dimension of the MPS
    d:          local Hilbert space dimension
    boundary:   determines the boundary condition for the MPS
    init:       Type of initial condition
    eps:        Each mps matrix is multiplied by eps
    use_biases: Determines if we use the biases
    normalize:  determines if we divide the output with its one norm
    init_diag:  Determines the initial values on the diagonal of the MPS
    ti:         Translationally invariant measurement
    """

    def __init__(
        self,
        nout,
        D=10,
        d=2,
        boundary="c",
        eps=1.0,
        use_biases=False,
        normalize=True,
        init_diag=1.0,
        name="L1_measurement",
        ti=False,
    ):
        super(MPSMeasurement, self).__init__(name=name)
        self.nout = nout
        self.D = D
        self.d = d
        self.boundary = boundary
        self.eps = eps
        self.use_biases = use_biases
        self.normalize = normalize
        self.init_diag = init_diag
        self.ti = ti

    def get_config(self):
        conf = {}
        conf["nout"] = self.nout
        conf["D"] = self.D
        conf["d"] = self.d
        conf["boundary"] = self.boundary
        conf["eps"] = self.eps
        conf["use_biases"] = self.use_biases
        conf["normalize"] = self.normalize
        conf["init_diag"] = self.init_diag
        conf["ti"] = self.ti
        conf["name"] = self.name
        return conf

    def get_mps(self, i):
        bias = None
        if self.ti:
            if i == 0:
                if self.use_biases:
                    bias = self.biases[0]
                return self.mps[0], bias
            if i == self.n-1:
                if self.use_biases:
                    bias = self.biases[2]
                return self.mps[2], bias
            if self.use_biases:
                bias = self.biases[1]
            return self.mps[1], bias
        else:
            if self.use_biases:
                bias = self.biases[i]
            return self.mps[i], bias

    def build(self, input_shape):
        """
        On the first run we construct the mps matrices.
        The shape of which is automatically determined from the input_shape
        """
        self.n = len(input_shape)
        self.iout = self.n // 2
        self.mps, self.biases, self.Aout = measurement_kernels(
            n=self.n,
            D=self.D,
            nout=self.nout,
            iout=self.iout,
            d=self.d,
            boundary=self.boundary,
            use_biases=self.use_biases,
            init_diag=self.init_diag,
            ti=self.ti
        )

    def call(self, input):
        alist = []
        n = self.n

        # Contracting both MPSs on each site to get local transfer matrices
        # Also adding the bias term at each site
        for i in range(n):
            mps, bias = self.get_mps(i)
            # Here we assume that the MPS has a bond dimension 1
            A = tf.einsum("...ijk,ljm->...ilkm", input[i], mps)
            if self.use_biases:
                A += bias
            if i == self.iout:
                A = tf.einsum("bij,...kjlm->...bkilm", self.Aout, A)
            alist.append(A)

        # Contracting neighbouring pairs of matrices until only one remains
        for i in range(int(np.log2(n)) + 1):
            blist = []
            for i in range(len(alist) // 2):
                A1 = alist[2 * i]
                A2 = alist[2 * i + 1]
                if len(A1.shape) == 6:
                    blist.append(tf.einsum("abijkl,aklmn->abijmn", A1, A2))
                elif len(A2.shape) == 6:
                    blist.append(tf.einsum("aijkl,abklmn->abijmn", A1, A2))
                else:
                    blist.append(tf.einsum("aijkl,aklmn->aijmn", A1, A2))

            if len(alist) % 2 == 1:
                blist.append(alist[-1])
            alist = blist
            if len(alist) == 1:
                break

        # Returning the trace of the remaining matrix
        res = tf.einsum("abijij->ab", alist[0])
        if self.boundary == "c":
            res = res / self.D

        if self.normalize:
            res = tf.abs(res)
            res = res / tf.reduce_sum(res, axis=1, keepdims=True)

        return res


def measurement_kernels(
    n,
    D,
    nout,
    iout,
    d=2,
    boundary="c",
    eps=1e-9,
    use_biases=False,
    dtype=tf.float32,
    trainable=True,
    init_diag=1.0,
    ti=False
):

    if ti:
        Dl = 1
        Dr = 1
        biases = []
        if boundary == "c":
            Dl = D
            Dr = D

        ashape = [Dl, d, D]
        Al = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
            init_diag * tf.eye(Dl, D, batch_shape=[d]), perm=[1, 0, 2])
        Al = tf.Variable(Al, trainable=trainable, dtype=dtype, name=f"Al")

        ashape = [D, d, D]
        A = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
            init_diag * tf.eye(D, D, batch_shape=[d]), perm=[1, 0, 2])
        A = tf.Variable(A, trainable=trainable, dtype=dtype, name=f"A")

        ashape = [D, d, Dr]
        Ar = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
            init_diag * tf.eye(D, Dr, batch_shape=[d]), perm=[1, 0, 2])
        Ar = tf.Variable(Ar, trainable=trainable, dtype=dtype, name=f"Ar")

        mps = [Al, A, Ar]

        if use_biases:
            bshape = [1, 1, Dl, 1, D]
            Bl = np.random.normal(loc=0, scale=eps, size=bshape)
            Bl[0, 0, :, 0, :] += np.eye(Dl, D)
            Bl = tf.Variable(Bl, dtype=dtype, name=f"Bl")

            bshape = [1, 1, D, 1, D]
            B = np.random.normal(loc=0, scale=eps, size=bshape)
            B[0, 0, :, 0, :] += np.eye(D, D)
            B = tf.Variable(B, dtype=dtype, name=f"B")

            bshape = [1, 1, D, 1, Dr]
            Br = np.random.normal(loc=0, scale=eps, size=bshape)
            Br[0, 0, :, 0, :] += np.eye(D, Dr)
            Br = tf.Variable(Br, dtype=dtype, name=f"Br")

            biases = [Bl, B, Br]

        Aout = tf.Variable(
            tf.eye(D, D, batch_shape=[nout]) +
            tf.random.normal([nout, D, D], mean=0, stddev=eps),
            trainable=trainable,
            dtype=dtype,
            name=f"Aout"
        )

        return mps, biases, Aout
    else:
        if boundary == "c":
            dims = (n + 1) * [D]
        elif boundary == "o":
            dims = [1] + [bond_dimension(D, d, n, i) for i in range(n)]
        else:
            logging.warning(
                "Unknown boundary condition. Using the periodic boundary condition")
            dims = (n + 1) * [D]
        mps = []
        biases = None
        Aout = None
        if use_biases:
            biases = []
        for i in range(n):
            dl = dims[i]
            dr = dims[i + 1]
            ashape = [dl, d, dr]
            A = tf.random.normal(ashape, mean=0, stddev=eps) + tf.transpose(
                init_diag * tf.eye(dl, dr, batch_shape=[d]), perm=[1, 0, 2])
            mps.append(tf.Variable(A, trainable=trainable,
                                   dtype=dtype, name=f"A{i}"))
            if i == iout:
                Aout = tf.Variable(
                    tf.eye(dl, dl, batch_shape=[nout]) +
                    tf.random.normal([nout, dl, dl], mean=0, stddev=eps),
                    trainable=trainable,
                    dtype=dtype,
                    name=f"Aout{i}"
                )
            if use_biases:
                bshape = [1, 1, dl, 1, dr]
                B = np.random.normal(loc=0, scale=eps, size=bshape)
                B[0, 0, :, 0, :] += np.eye(dl, dr)
                biases.append(tf.Variable(B, dtype=dtype), name=f"B{i}")

        return mps, biases, Aout


class MPS(tf.keras.layers.Layer):
    def __init__(self, D=10, d=2, C=10, stddev=1e-5, ti=False, init_diag=1.0, **kwargs):
        super(MPS, self).__init__(**kwargs)
        self.D = D
        self.d = d
        self.C = C
        self.ti = ti
        self.stddev = stddev
        self.init_diag = init_diag
        self.config = {
            "D": D,
            "d": d,
            "C": C,
            "ti": ti,
            "stddev": stddev,
            "init_diag": init_diag,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def build(self, input_shape):
        # We assume the input_shape is (N,nbatch,d)
        N = input_shape[0]
        d = input_shape[2]
        C = self.C
        dtype = tf.float32
        assert d == self.d, f"Input shape should be (N,nbatch,d). Obtained feature size d={d}, expected {self.d}."

        self.n = N
        stddev = self.stddev
        D = self.D
        init_diag = self.init_diag

        self.Aout = tf.Variable(tf.eye(D, D, batch_shape=[C]) + tf.random.normal(
            shape=(C, D, D), stddev=stddev), dtype=dtype, name="tensor", trainable=True)
        if self.ti:
            A = tf.random.normal(shape=(D, D, d), mean=0, stddev=stddev) + tf.transpose(
                init_diag * tf.eye(D, D, batch_shape=[d]), perm=[2, 1, 0])
            self.tensor = tf.Variable(
                A, trainable=True, dtype=dtype, name=f"tensor")

            Bl = tf.reshape(init_diag * tf.eye(D, 1),
                            [-1]) + tf.random.normal(shape=(D), stddev=stddev)
            self.Bl = tf.Variable(Bl, name="left_boundary",
                                  dtype=dtype, trainable=True)
            Bl = tf.reshape(init_diag * tf.eye(D, 1),
                            [-1]) + tf.random.normal(shape=(D), stddev=stddev)
            self.Br = tf.Variable(Br, name="right_boundary",
                                  dtype=dtype, trainable=True)
        else:
            A = tf.random.normal(shape=(N, D, D, d), mean=0, stddev=stddev) + tf.transpose(
                init_diag * tf.eye(D, D, batch_shape=[N, d]), perm=[0, 3, 2, 1])
            self.tensor = tf.Variable(
                A, name="tensor", dtype=dtype, trainable=True)

    def call(self, input, mode=None):
        # returns the log-overlap
        d = self.d
        n = input.shape[0]
        assert d == self.d, f"Input shape should be (N,nbatch,d). Obtained feature size d={d}, expected {self.d}."
        # assert n == self.n, f"Input shape should be (N,nbatch,d). Obtained input size N={n}, expected {self.n}."

        if self.ti:
            A = tf.einsum("nbi,lri->nblr", input, self.tensor)
        else:
            A = tf.einsum("nbi,nlri->nblr", input, self.tensor)

        if mode is None:  # If no mode is set we check the conditions automatically on the fly
            D = self.D
            if D**2*n < np.log2(1.0*n)*D**3:
                mode = "sequential"
            else:
                mode = "parallel"
        nhalf = n//2
        if mode == "sequential":
            if self.ti:
                Al = tf.einsum("l,blr->br", self.Bl, A[0])
            else:
                Al = A[0, :, 0, :]
            for i in range(1, nhalf):
                Al = tf.einsum("bl,blr->br", Al, A[i])

            if self.ti:
                Ar = tf.einsum("r,blr->bl", self.Br, A[n-1])
            else:
                Ar = A[n-1, :, :, 0]
            for i in range(n-2, nhalf-1, -1):
                Ar = tf.einsum("blr,br->bl", A[i], Ar)

        else:
            # Contracting neighbouring pairs of matrices until only one remains (left side)
            Al = parallel_contract(A[:nhalf])[:, 0, :]
            # Contracting neighbouring pairs of matrices until only one remains (right side)
            Ar = parallel_contract(A[nhalf:])[:, :, 0]

        Aout = tf.einsum("bl,olr->bor", Al, self.Aout)
        out = tf.einsum("bor,br->bo", Aout, Ar)
        return out


def parallel_contract(X):
    alist = X
    n = alist.shape[0]
    nalist = alist.shape[0]
    Alast = None
    for i in range(int(np.log2(n)) + 1):
        indo = np.arange(0, 2*(nalist//2), 2)
        inde = np.arange(1, 2*(nalist//2), 2)
        Ao = tf.gather(alist, indo, axis=0)
        Ae = tf.gather(alist, inde, axis=0)
        alist = tf.einsum("nblk,nbkr->nblr", Ao, Ae)
        if nalist % 2 == 1:
            if Alast is not None:
                Alast = tf.einsum("blk,bkr->blr", alist[-1], Alast)
            else:
                Alast = alist[-1]
        nalist = alist.shape[0]
        if alist.shape[0] == 1:
            if Alast is not None:
                Alast = tf.einsum("blk,bkr->blr", alist[0], Alast)
            else:
                Alast = alist[0]
            break
    return Alast

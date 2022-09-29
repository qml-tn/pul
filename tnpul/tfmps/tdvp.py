import tensorflow as tf
from tnpul.utils.mps import bond_dimension
import numpy as np
import tensorflow_probability as tfp


class MPS_TDVP(tf.keras.layers.Layer):
    """Layer that converts the image input into an MPS
    d:          local Hilbert space dimension: 2(default)
    D:          bond dimension of the MPO: 2(default)
    out_norm:   Normalization of the output vector: "none", "L1" (default), "L2"
    activation: Activation function applied after the local affine transformation
                Possible options: "none", relu(default), expm, exp
    mode:       Parallel or sequential evaluation of the local hamiltonians: parallel(default), sequential
    eps:        Determines the standard deviation of the initializations: 1e-6(default)
    ti:         Determines if the hamiltonian is translationaly invariant: False (default)
    return_mps: If enabled we return an mps instead of the embedding vectors. 0 (default-disabled), 1 (enabled) 
    """

    def __init__(self, d=2, D=2, out_norm="L1", activation="relu", mode="parallel", eps=1e-6, ti=False, name="TDVP_D1", return_mps=1):
        super(MPS_TDVP, self).__init__(name=name)
        self.d = d
        self.D = D
        self.eps = eps
        self.out_norm = out_norm
        self.activation = activation
        self.mode = mode
        self.ti = ti
        self.return_mps = return_mps
        self.config = {
            "d": d,
            "D": D,
            "eps": eps,
            "out_norm": out_norm,
            "activation": activation,
            "mode": mode,
            "ti": ti,
            "name": name,
            "return_mps": return_mps,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def build(self, input_shape):
        """
        The first dimension is the batch_size. 
        We assume here that the MPS has bond dimension 1. 
        Higher bond dimensions come with complications due to gauge invariance, which is 
        trivial in the D=1.
        TODO: Extend the layer to larger bond dimensions.
        """
        if self.return_mps:
            n = len(input_shape)
            self.n = n
            for i in range(n):
                dims = input_shape[0]
                assert dims[-3] == 1, f"Left bond dimension of the MPS at position {i} is {dims[-3]}. Maximum is 1."
                assert dims[-1] == 1, f"Right bond dimension of the MPS at position {i} is {dims[-1]}. Maximum is 1."
        else:
            n = input_shape[0]
            self.n = n

        self.flatten = tf.keras.layers.Flatten()
        self.mpo = mpo_kernel(self.n, self.D, self.d,
                              self.activation, self.eps, ti=self.ti)

        m = int(np.log(n-1)/np.log(2.0))+1
        self.nbin = m

        self.positions = np.zeros([m, m], dtype=int)
        for i in range(0, m-1):
            k = 2
            for j in range(i+1, m):
                self.positions[i, j] = k
                k = 2*k

    def get_binary(self, i):
        return [int(c) for c in format(i, f'0{self.nbin}b')]

    def get_mpo(self, i):
        d = self.d
        D = self.D
        pi_half = np.pi/2.0
        if self.ti:
            alist = []
            for mpo in self.mpo:
                alist.append(tfp.math.fill_triangular(
                    tf.math.atan(mpo)/pi_half, upper=True))
            A = tf.transpose(tf.reshape(tf.stack(alist), [
                             d, d, D, D]), perm=[2, 0, 1, 3])
            if i == 0:
                return A[:1, :, :, :]
            if i == self.n-1:
                return A[:, :, :, -1:]
            return A
        else:
            return self.mpo[i]

    def call(self, input):
        n = self.n
        # input.shape = (N, nbatch, d)
        # nbatch = input[0].shape[0]
        # Contractions with the state

        Alist = []
        for i in range(n):
            X = tf.reshape(input[i], [-1, self.d])
            mpo = self.get_mpo(i)
            # We use L2 normalized vectors for the TDVP
            X = tf.math.divide(X, tf.linalg.norm(X, axis=1, keepdims=True))
            A = tf.einsum("ai,jikl,ak->ajl", X, mpo, X)
            Alist.append(A)

        M0 = tf.ones([1, 1])
        if self.mode == "parallel":
            # Parallel computation of the local hamiltonian: complexity=log2(n)d^3.
            # Left contractions
            Bleft = [Alist]
            for i in range(self.nbin-1):
                blist = []
                for i in range(len(Alist) // 2):
                    A1 = Alist[2 * i]
                    A2 = Alist[2 * i + 1]
                    blist.append(tf.einsum("aij,ajk->aik", A1, A2))
                Alist = blist
                Bleft.append(blist)
                if len(Bleft) == 1:
                    break

            Aleft = []
            for i in range(n):
                Al = None
                x = self.get_binary(i)
                xp = x@self.positions
                for j in range(self.nbin):
                    if x[j]:
                        A = Bleft[self.nbin-1-j][xp[j]]
                        if Al is None:
                            Al = A[:, 0, :]
                        else:
                            Al = tf.einsum("ai,aij->aj", Al, A)
                if Al is not None:
                    Aleft.append(Al)

            # Right contractions
            Alist = Bleft[0]
            Bright = [Alist]
            for i in range(self.nbin-1):
                blist = []
                for i in range(len(Alist) // 2):
                    A1 = Alist[-2 * i - 2]
                    A2 = Alist[-2 * i - 1]
                    blist = [tf.einsum("aij,ajk->aik", A1, A2)] + blist
                Alist = blist
                Bright.append(blist)
                if len(Bright) == 1:
                    break

            Aright = []
            for i in range(n):
                # Ar = M0
                Ar = None
                x = self.get_binary(i)
                xp = x@self.positions
                for j in range(self.nbin):
                    if x[j]:
                        A = Bright[self.nbin-1-j][-1-xp[j]]
                        if Ar is None:
                            Ar = A[:, :, 0]
                        else:
                            Ar = tf.einsum("aij,aj->ai", A, Ar)
                if Ar is not None:
                    Aright = [Ar] + Aright
        else:
            # Sequentially calculating the left and the right Hamiltonians
            # Left part of the Hamiltonian
            Aleft = []
            for i in range(n - 1):
                if i == 0:
                    A = Alist[i][:, 0, :]
                else:
                    A = tf.einsum("ai,aij->aj", Aleft[-1], Alist[i])
                Aleft.append(A)

            # Right part of the Hamiltonian
            Aright = []
            for i in range(n - 1):
                if i == 0:
                    A = Alist[n-1-i][:, :, 0]
                else:
                    A = tf.einsum("aij,aj->ai", Alist[n - 1 - i], Aright[0])
                Aright = [A] + Aright

        # Calculate the local map
        Mlist = []
        for i in range(n):
            mpo = self.get_mpo(i)
            if i == 0:
                H = tf.einsum(
                    "jkl,al->ajk", mpo[0, :, :, :], Aright[i])
            elif i == n-1:
                H = tf.einsum(
                    "ai,ijk->ajk", Aleft[i-1], mpo[:, :, :, 0])
            else:
                # H = tf.einsum("ai,ijkl,al->ajk", Aleft[i-1], mpo, Aright[i])
                H = tf.einsum("al,ijkl->ajki", Aright[i], mpo)
                H = tf.einsum("ai,ajki->ajk", Aleft[i-1], H)
            if self.activation == "expm":
                if self.d == 2:
                    b = (H[:, 0, 1]-H[:, 1, 0])/2.0
                    # This explicit calculation should improve the calculation of the matrix exponential...
                    H = tf.reshape(tf.stack([tf.math.cos(b), tf.math.sin(
                        b), -tf.math.sin(b), tf.math.cos(b)], axis=-1), [-1, 2, 2])
                else:
                    # For larger system sizes we have to calculate the exponential numerically
                    H = tf.linalg.expm((H-tf.transpose(H))/2.)
            Mlist.append(H)

        # Calculating the new mps
        mps = []
        for i in range(n):
            A = tf.einsum(
                "aij,aj->ai",
                Mlist[i]/n,
                tf.reshape(input[i], [-1, self.d]),
            )
            if self.activation == "relu":
                # The slope below the threshold should not be 0 in order
                # to avoid dividing by 0 in the next step if normalization is used
                A = tf.keras.activations.relu(A, alpha=-0.25)
            if self.activation == "exp":
                A = tf.keras.activations.exponential(A)

            # Final normalization of the vector
            if self.out_norm == "L2":
                # A = A / tf.linalg.norm(A, axis=1, keepdims=True)
                A = tf.math.divide(A, tf.linalg.norm(A, axis=1, keepdims=True))
            elif self.out_norm == "L1":
                # We use also absolute value in order to avoid dividing by 0
                # and to get a value consistent with the linear embedding.
                # A = A / tf.math.reduce_sum(A, axis=1, keepdims=True)
                A = tf.math.abs(A)
                A = tf.math.divide(A, tf.math.reduce_sum(
                    A, axis=1, keepdims=True))
            if self.return_mps:
                mps.append(tf.reshape(A, [-1, 1, self.d, 1]))
            else:
                mps.append(tf.reshape(A, [-1, self.d]))
        if self.return_mps:
            return mps
        else:
            return tf.stack(mps, axis=0)


def get_init_A(d, D, diag=1e1, trainable=True, eps=1e-6, dtype=tf.float32):
    inds = range(int(D*(D+1)/2))
    inds_mat = tfp.math.fill_triangular(inds, upper=True).numpy()

    alist = []

    for i in range(d*d):
        v = (np.random.rand(len(inds))-0.5)*eps
        if np.mod(i, d+1) == 0:
            v[np.diag(inds_mat)] += 1e2
            v[inds_mat[0, d-1]] += diag
        v = tf.Variable(v, trainable=trainable, dtype=dtype)
        alist.append(v)
    return alist


def mpo_kernel(n, D, d=2, activation="relu", eps=1e-9, dtype=tf.float32, trainable=True, ti=False):
    dims = [1] + [bond_dimension(D, d * d, n, i) for i in range(n)]

    mps = []

    diag = 1e2
    if ti:
        if activation == "expm":
            diag = 0.0
        return get_init_A(d, D, diag=diag, trainable=trainable, eps=eps)
    else:
        for i in range(n):
            dl = dims[i]
            dr = dims[i + 1]
            ashape = [dl, d, d, dr]
            A = tf.random.normal(ashape, mean=0, stddev=eps)
            if activation != "expm":
                ID = np.zeros(ashape)
                ID[0, :, :, 0] = np.eye(d, d) * diag
                A += ID
            mps.append(tf.Variable(A, trainable=trainable, dtype=dtype))

    return mps


class TDVP_PS(tf.keras.layers.Layer):
    """TDVP layer for product states/embeddings
    d:          local Hilbert space dimension: 2(default)
    D:          bond dimension of the MPO: 2(default)
    out_norm:   Normalization of the output vector: "none", "L1" (default), "L2"
    activation: Activation function applied after the local affine transformation
                Possible options: "none", relu(default), expm
    stddev:     Determines the standard deviation of the initializations: 1e-6(default)
    use_mask:       Enables/disables the masking of the MPS tensors (lower-triangular). 0 (mask disabled), 1 (mask enabled - default) 
    ti:         Determines if the hamiltonian is translationaly invariant: True (default)
    """

    def __init__(self, d=2, D=2, out_norm="L1", activation="relu", stddev=1e-6, ti=True, name="TDVP", use_mask=True, kernel_activation=False, residual=False, nrm_lambda=1.0, dtype=np.float32):
        super(TDVP_PS, self).__init__(name=name, dtype=dtype)
        self.d = d
        self.D = D
        self.stddev = stddev
        self.out_norm = out_norm
        self.activation = activation
        self.use_mask = use_mask
        self.ti = ti
        self.kernel_activation = kernel_activation
        self.residual = residual
        self.nrm_lambda = nrm_lambda
        if self.residual:
            self.res_coef = tf.Variable(1.0, dtype=dtype, trainable=True)
        self.config = {
            "d": d,
            "D": D,
            "stddev": stddev,
            "out_norm": out_norm,
            "activation": activation,
            "use_mask": use_mask,
            "ti": ti,
            "kernel_activation": kernel_activation,
            "residual": residual,
            "nrm_lambda": nrm_lambda,
            "name": name,
        }

    def get_config(self):
        return self.config

    def build(self, input_shape):  # (n,bs,d)
        dtype = self.dtype
        n, _, d = input_shape
        assert d == self.d, f"Feature dimension d={d} but expected {self.d}."
        self.n = n
        D = self.D
        stddev = self.stddev

        diag_const = 1.0
        if self.kernel_activation:
            diag_const = 1.e3

        self.scale = tf.Variable(1, trainable=True, dtype=dtype)

        if self.ti:
            mpo = np.random.normal(scale=stddev, size=[D, D, d, d])
            # mpo = np.tan(np.random.normal(scale=stddev, size=[D, D, d, d]))
            for i in range(d):
                mpo[:, :, i, i] += np.eye(D) * diag_const
            if self.activation != "expm":
                mpo[0, -1, :, :] += diag_const * np.eye(d)
        else:
            mpo = np.random.normal(scale=stddev, size=[n, D, D, d, d])
            # mpo = np.tan(np.random.normal(scale=stddev, size=[n, D, D, d, d]))
            for i in range(d):
                mpo[:, :, :, i, i] += np.eye(D) * diag_const
            if self.activation != "expm":
                mpo[:, 0, -1, :, :] += diag_const * np.eye(d)

        self.mpo = tf.Variable(mpo, trainable=True, dtype=dtype)

        if self.use_mask:
            if self.ti:
                mask = np.ones([D, D, d, d])
                for i in range(D):
                    for j in range(i):
                        mask[i, j] = 0
            else:
                mask = np.ones([n, D, D, d, d])
                for i in range(D):
                    for j in range(i):
                        mask[:, i, j, :, :] = 0
            self.mask = tf.constant(mask, dtype=dtype)

        vl = np.zeros([1, D])
        vl[0, 0] = 1.0
        self.Bl = tf.Variable(vl, dtype=dtype, trainable=True)

        vr = np.zeros([1, D])
        vr[0, -1] = 1.0
        self.Br = tf.Variable(vr, dtype=dtype, trainable=True)

        # positions and nbins for parallel computation
        m = int(np.log(n-1)/np.log(2.0))+1
        self.nbin = m

        self.positions = np.zeros([m, m], dtype=int)
        for i in range(0, m-1):
            k = 2
            for j in range(i+1, m):
                self.positions[i, j] = k
                k = 2*k

    def call(self, input):  # (n,bs,d)
        n, _, d = input.shape
        # n = tf.shape(input)[0] # We determine on the fly the size of the input
        # bs is a dynamic variable hence we have to use tf.shape
        bs = tf.shape(input)[1]
        # d = tf.shape(input)[2]
        assert d == self.d, f"Feature dimension d={d} but expected {self.d}."
        if not self.ti:
            assert n == self.n, f"Feature size n={n} but expected {self.n}."

        if self.kernel_activation:
            mpo = 2.0*tf.math.atan(self.mpo)/np.pi
        else:
            mpo = self.mpo
        if self.use_mask:
            mpo = mpo*self.mask

        # We assume that the input is L2 normalized
        # x = input/tf.linalg.norm(input, axis=2, keepdims=True)

        if self.ti:
            mpoc = tf.einsum("nbi,lrij->nblrj", input, mpo)
        else:
            mpoc = tf.einsum("nbi,nlrij->nblrj", input, mpo)
        mpoc = tf.einsum("nblrj,nbj->nblr", mpoc, input)

        Bl = tf.repeat(self.Bl, repeats=bs, axis=0)
        Als = [Bl]
        Br = tf.repeat(self.Br, repeats=bs, axis=0)
        Ars = [Br]
        parallel = True
        if np.log2(n) % 2 == 0 and parallel:
            Alist = parallel_contract_list(mpoc)
            als_bulk = contractions_left(Alist, self.positions, self.nbin)
            Als = Als + als_bulk
            ars_bulk = contractions_right(Alist, self.positions, self.nbin)
            Ars = ars_bulk + Ars
        else:
            for i in range(0, n-1):
                Al = tf.einsum("bl,blr->br", Als[-1], mpoc[i])
                Al = Al/tf.linalg.norm(Al)
                Als.append(Al)
            for i in range(n-1, 0, -1):
                Ar = tf.einsum("blr,br->bl", mpoc[i], Ars[0])
                Ar = Ar/tf.linalg.norm(Ar)
                Ars = [Ar] + Ars

        Als = tf.stack(Als, axis=0)
        Ars = tf.stack(Ars, axis=0)

        if self.ti:
            Hl = tf.einsum("nbl,lrij->nbijr", Als, mpo)
        else:
            Hl = tf.einsum("nbl,nlrij->nbijr", Als, mpo)
        H = tf.einsum("nbijr,nbr->nbij", Hl, Ars)

        H = H * self.scale

        # tf.print(H[0, 0])

        # The input is not normalized for the application of the HMPO
        if self.activation == "expm":
            raise Exception("Not yet implemented")
        else:
            out = tf.einsum("nbij,nbj->nbi", H, input)

        if self.residual:
            out += self.res_coef*input

        if self.activation == "relu":
            out = tf.keras.activations.relu(out, alpha=-0.125)

        if self.activation == "elu":
            out = tf.keras.activations.elu(out)

        if self.activation == "sigmoid":
            out = tf.keras.activations.sigmoid(out)

        if self.activation == "square":
            out = out*out

        # This regularization helps with the stability of a deep network.
        # It guarantees that the output is almost L2 normalized even without explicit normalization afterwards.
        if self.nrm_lambda > 0:
            reg_loss = tf.identity(
                self.nrm_lambda*tf.reduce_mean(tf.math.abs(tf.linalg.norm(out, axis=2)-1.)), name="out_norm")
            self.add_loss(reg_loss)

        if self.out_norm == "L2":
            out = out/tf.linalg.norm(out, axis=2, keepdims=True)
        elif self.out_norm == "L1":
            out = out/tf.reduce_sum(out, axis=2, keepdims=True)

        if self.nrm_lambda > 0:
            reg_loss = tf.identity(-tf.clip_by_value(self.nrm_lambda*tf.reduce_mean(
                tf.math.abs(out[:, :, 0]-out[:, :, 1])),0,1), name="out_diff")
            self.add_loss(reg_loss)

        return out


def parallel_contract_list(X):
    A = X
    n = A.shape[0]
    nA = A.shape[0]
    alist = [A]
    for i in range(int(np.log2(n))):
        indo = np.arange(0, 2*(nA//2), 2)
        inde = np.arange(1, 2*(nA//2), 2)
        Ao = tf.gather(A, indo, axis=0)
        Ae = tf.gather(A, inde, axis=0)
        A = tf.einsum("nblk,nbkr->nblr", Ao, Ae)
        nA = A.shape[0]
        A = A / tf.linalg.norm(A)
        alist.append(A)
        if A.shape[0] == 1:
            break
    return alist


def contractions_left(Alist, positions, nbin):
    n = Alist[0].shape[0]
    Aleft = []
    for i in range(n):
        Al = None
        x = get_binary(i, nbin=nbin)
        xp = x@positions
        for j in range(nbin):
            if x[j]:
                A = Alist[nbin-1-j][xp[j]]
                if Al is None:
                    Al = A[:, 0, :]
                else:
                    Al = tf.clip_by_value(
                        tf.einsum("al,alr->ar", Al, A), clip_value_min=-1e3, clip_value_max=1e3)
        Al = Al / tf.linalg.norm(Al)
        if Al is not None:
            Aleft.append(Al)
    return Aleft


def contractions_right(Alist, positions, nbin):
    n = Alist[0].shape[0]
    assert np.log2(
        n) % 2 == 0, f"Number of elements should be a power of 2, but got {n}."
    Aright = []
    for i in range(n):
        # Ar = M0
        Ar = None
        x = get_binary(i, nbin=nbin)
        xp = x@positions
        for j in range(nbin):
            if x[j]:
                A = Alist[nbin-1-j][-1-xp[j]]
                if Ar is None:
                    Ar = A[:, :, -1]
                else:
                    Ar = tf.clip_by_value(
                        tf.einsum("alr,ar->al", A, Ar), clip_value_min=-1e3, clip_value_max=1e3)
        Ar = Ar / tf.linalg.norm(Ar)
        if Ar is not None:
            Aright = [Ar] + Aright
    return Aright


def get_binary(i, nbin):
    return [int(c) for c in format(i, f'0{nbin}b')]

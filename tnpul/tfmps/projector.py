import numpy as np
import tensorflow as tf
from tnpul.tfmps.embeddings import FourierEmbedding


class Projector(tf.keras.layers.Layer):
    def __init__(self, D, d=2, S=2, stddev=0.5, dropout=0):
        super(Projector, self).__init__()
        self.D = D
        self.S = S
        self.d = d
        self.stddev = stddev
        self.dropout = dropout
        self.config = {
            "D": D,
            "d": d,
            "S": S,
            "stddev": stddev,
            "dropout": dropout,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def build(self, input_shape):
        # We assume the input_shape is (N,nbatch, feature size=2)
        N = input_shape[0]
        self.N = N
        S = self.S
        stddev = self.stddev
        self.Nout = int(np.ceil((N-1)/S)+1)
        self.Nin = self.N - self.Nout
        D = self.D
        self.MPS_input = tf.Variable(tf.random.normal(shape=(
            self.Nin, D, D, self.d), stddev=stddev), name="mps_input", trainable=True)
        self.MPS_output = tf.Variable(tf.random.normal(shape=(
            self.Nout, D, D, self.d, self.d), stddev=stddev), name="mps_output", trainable=True)

    def log_norm(self):
        n = self.N
        S = self.S
        D = self.D
        Al = np.zeros([D, D])
        Al[0, 0] = 1
        Al = tf.constant(Al, dtype=tf.float32)
        scals = []
        i_in = 0
        for i in range(n):
            amax = tf.reduce_max(tf.math.abs(Al))
            Al = Al / amax
            scals.append(amax)
            # tf.print(tf.linalg.norm(Al))
            if (i % S == 0 or i == n-1):
                if i == n-1:
                    M = self.MPS_output[self.Nout-1]
                else:
                    M = self.MPS_output[int(i/S)]
                Al = tf.einsum("du,dkij->kuij", Al, M)
                Al = tf.einsum("kuij,ulij->kl", Al, M)
            else:
                M = self.MPS_input[i_in]
                Al = tf.einsum("du,dki->kui", Al, M)
                Al = tf.einsum("kui,uli->kl", Al, M)
                i_in += 1
        scals.append(Al[0, 0])
        lnrm = tf.reduce_sum(tf.math.log(tf.stack(scals)))
        return lnrm

    def sample(self, local_sampler, rep=1):
        n = self.N
        L = n//rep
        d = self.d
        sampQ = np.zeros(n)
        samp = np.zeros([n, d])

        sample = []
        for i in range(L):
            # x,emb = self.local_sample(i,samp,sampQ,rep,local_sampler)
            rholist = self.get_rholist(i, samp, sampQ, rep=rep)
            x, emb = local_sampler(rholist)
            sample.append(x)
            for j in range(rep):
                samp[i+L*j, :] = emb[j].numpy()
                sampQ[i+L*j] = 1
        return sample

    def get_rholist(self, i, samp, sampQ, rep=1):
        n = self.N
        L = n//rep
        d = self.d
        pi_half = np.pi/2.

        assert i < L, "Index should be smaller than the input length without repetitions."

        rholist = []
        for l in range(rep):
            rho0 = self.get_rho_local(i+l*L, samp, sampQ)
            rho0 = rho0/tf.linalg.norm(rho0)
            rholist.append(rho0)
        return rholist

    def get_rho_local(self, i, samp, sampQ):
        n = self.N
        S = self.S

        Al = self.get_Al(i-1, samp, sampQ)
        Ar = self.get_Ar(i+1, samp, sampQ)

        if (i % S == 0 or i == n-1):
            if i == n-1:
                M = self.MPS_output[self.Nout-1]
            else:
                M = self.MPS_output[int(i/S)]

            rho = tf.einsum("du,dkai->kuai", Al, M)
            rho = tf.einsum("kuai,ulbi->klab", rho, M)
            rho = tf.einsum("kl,klab->ab", Ar, rho)
        else:
            i_in = i-i//S-1
            M = self.MPS_input[i_in]
            rho = tf.einsum("du,dka->kua", Al, M)
            rho = tf.einsum("kua,ulb->klab", rho, M)
            rho = tf.einsum("kl,klab->ab", Ar, rho)

        return rho

    def get_Al(self, k, samp, sampQ):
        assert len(samp) == len(
            sampQ), "Arrays samp and sampQ should have the same length"
        S = self.S
        D = self.D
        n = self.N
        Al = np.zeros([D, D])
        Al[0, 0] = 1
        Al = tf.constant(Al, dtype=tf.float32)
        scals = []
        i_in = 0
        for i in range(k+1):
            amax = tf.reduce_max(tf.math.abs(Al))
            # Al = Al / amax
            Al = tf.math.divide_no_nan(Al, amax)
            scals.append(amax)
            # tf.print(tf.linalg.norm(Al))
            if (i % S == 0 or i == n-1):
                if i == n-1:
                    M = self.MPS_output[self.Nout-1]
                else:
                    M = self.MPS_output[int(i//S)]
                if sampQ[i]:
                    M = tf.einsum("lrij,i->lrj", M, samp[i])
                    Al = tf.einsum("du,dkj->kuj", Al, M)
                    Al = tf.einsum("kuj,ulj->kl", Al, M)
                else:
                    Al = tf.einsum("du,dkij->kuij", Al, M)
                    Al = tf.einsum("kuij,ulij->kl", Al, M)
            else:
                M = self.MPS_input[i_in]
                if sampQ[i]:
                    M = tf.einsum("lri,i->lr", M, samp[i])
                    Al = tf.einsum("du,dk->ku", Al, M)
                    Al = tf.einsum("ku,ul->kl", Al, M)
                else:
                    Al = tf.einsum("du,dki->kui", Al, M)
                    Al = tf.einsum("kui,uli->kl", Al, M)
                i_in += 1
        return Al

    def get_Ar(self, k, samp, sampQ):
        assert len(samp) == len(
            sampQ), "Arrays samp and sampQ should have the same length"
        S = self.S
        D = self.D
        n = self.N
        Ar = np.zeros([D, D])
        Ar[0, 0] = 1
        Ar = tf.constant(Ar, dtype=tf.float32)
        scals = []
        i_in = self.MPS_input.shape[0]-1
        for i in range(n-1, k-1, -1):
            amax = tf.reduce_max(tf.math.abs(Ar))
            # Ar = Ar / amax
            Ar = tf.math.divide_no_nan(Ar, amax)
            scals.append(amax)
            # tf.print(tf.linalg.norm(Al))
            if (i % S == 0 or i == n-1):
                if i == n-1:
                    M = self.MPS_output[self.Nout-1]
                else:
                    M = self.MPS_output[int(i//S)]
                if sampQ[i]:
                    M = tf.einsum("lrij,i->lrj", M, samp[i])
                    Ar = tf.einsum("du,kdj->kuj", Ar, M)
                    Ar = tf.einsum("kuj,luj->kl", Ar, M)
                else:
                    Ar = tf.einsum("du,kdij->kuij", Ar, M)
                    Ar = tf.einsum("kuij,luij->kl", Ar, M)
            else:
                M = self.MPS_input[i_in]
                if sampQ[i]:
                    M = tf.einsum("lri,i->lr", M, samp[i])
                    Ar = tf.einsum("du,kd->ku", Ar, M)
                    Ar = tf.einsum("ku,lu->kl", Ar, M)
                else:
                    Ar = tf.einsum("du,kdi->kui", Ar, M)
                    Ar = tf.einsum("kui,lui->kl", Ar, M)
                i_in -= 1
        return Ar

    def call(self, input, training):
        if training:
            drop = tf.cast(tf.random.uniform(
                [self.D]) > self.dropout, dtype=tf.float32)

        eps = tf.constant(1e-60)
        # returns the log-overlap
        S = self.S
        n = self.N
        A = tf.einsum("bi,lrij->blrj",
                      input[0], self.MPS_output[0, :1, :, :, :])
        if training:
            A = tf.einsum("blrj,r->blrj", A, drop)
        mps = [A]
        # Al = tf.eye(self.D,batch_shape=[1])
        i_in = 0
        Al = None
        out_S = []
        for i in range(1, n-1):
            if i % S == 0:
                x = tf.einsum("bi,lrij->blrj",
                              input[i], self.MPS_output[int(i/S)])
                A = tf.einsum("blr,brdi->bldi", Al, x)
                if training:
                    A = tf.einsum("bldi,d->bldi", A, drop)
                # Al = tf.eye(self.D,batch_shape=[1])
                Al = None
                mps.append(A)
            else:
                A = tf.einsum("bi,lri->blr", input[i], self.MPS_input[i_in])
                if training:
                    A = tf.einsum("blr,r->blr", A, drop)
                if Al is None:
                    Al = A
                else:
                    Al = tf.einsum("blr,brd->bld", Al, A)
                amax = tf.reduce_max(tf.math.abs(Al), axis=[
                                     1, 2], keepdims=True)
                # Al = Al/amax
                Al = tf.math.divide_no_nan(Al, amax)
                out_S.append(tf.reshape(amax, [-1]))
                i_in += 1
        x = tf.einsum("bi,lrij->blrj",
                      input[n-1], self.MPS_output[-1, :, :1, :, :])
        if Al is not None:
            A = tf.einsum("blr,brdi->bldi", Al, x)
        else:
            A = x
        mps.append(A)
        out_S = 2*tf.reduce_sum(tf.math.log(
            tf.stack(out_S, axis=-1)+eps), axis=1)

        # n = len(mps)
        out = []
        A = mps[0][:, 0, :, :]
        Al = tf.einsum("bui,bdi->bud", A, A)
        amax = tf.reduce_max(tf.math.abs(Al), axis=[1, 2], keepdims=True)
        # Al = Al/amax
        Al = tf.math.divide_no_nan(Al, amax)
        out.append(tf.reshape(amax, [-1]))
        for A in mps[1:]:
            # tf.print(tf.linalg.norm(Al,axis=[1,2]))
            Al = tf.einsum("bdu,bdli->blui", Al, A)
            Al = tf.einsum("blui,buri->blr", Al, A)
            amax = tf.reduce_max(tf.math.abs(Al), axis=[1, 2], keepdims=True)
            # Al = Al/amax
            Al = tf.math.divide_no_nan(Al, amax)
            out.append(tf.reshape(amax, [-1]))
        out.append(tf.math.abs(Al[:, 0, 0]))
        out = tf.reduce_sum(tf.math.log(tf.stack(out, axis=-1) + eps), axis=1)

        # tf.print(tf.math.reduce_max(tf.math.abs(out)),tf.math.reduce_max(tf.math.abs(out_S)))
        return out + out_S


def sample(L, d, models, c=0):
    m = len(models)
    sampQs = []
    samps = []
    blist = []

    for model in models:
        rep = model.repeat
        n = L*rep
        sampQ = np.zeros(n)
        samp = np.zeros([n, d])
        sampQs.append(sampQ)
        samps.append(samp)
        blist.extend(model.basis[:rep])

    local_sampler = Local_sampler(blist, d=d)

    sample = []
    for i in range(L):
        rholist = []
        for j in range(m):
            model = models[j]
            rep = model.repeat
            sampQ = sampQs[j]
            samp = samps[j]
            if c == 1:
                rholist.extend(model.mpop.get_rholist(i, samp, sampQ, rep=rep))
            else:
                rholist.extend(model.mpon.get_rholist(i, samp, sampQ, rep=rep))

        x, emb = local_sampler(rholist)
        sample.append(x)

        ib = 0
        for k in range(m):
            model = models[k]
            rep = model.repeat
            sampQ = sampQs[k]
            samp = samps[k]
            for j in range(rep):
                samp[i+L*j, :] = emb[ib].numpy()
                sampQ[i+L*j] = 1
                ib += 1
    return sample, sampQs

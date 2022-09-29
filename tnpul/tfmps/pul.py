import numpy as np
import tensorflow as tf
from tnpul.tfmps.projector import Projector
from tnpul.tfmps.embeddings import FourierEmbedding
from tnpul.tfmps.utils import Local_sampler


def PULloss(beta1=1.0, beta2=1.0, beta3=1.0, beta4=1.0, beta5=1.0, beta6=1.0, beta7=1.0, beta7all=0, beta8=1.0, leps=np.log(1.1), logr_pos=5, logr_neg=50):

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def pul_loss(y_true, y_pred):

        loss = 0
        y_label = tf.cast(y_true[:, 0], dtype=tf.float32)

        pos_bool = (y_pred[:, 0]-y_pred[:, 1]) > leps
        neg_bool = (y_pred[:, 1]-y_pred[:, 0]) > leps

        pos_full = tf.cast(pos_bool, dtype=tf.float32)
        neg_full = tf.cast(neg_bool, dtype=tf.float32)

        pos = pos_full * (1.0-y_label)
        neg = neg_full * (1.0-y_label)

        # positive labeled examples should be in the negetive kernel and positive unit sphere
        if beta1 > 0:
            loss_lp = beta1 * \
                tf.math.reduce_mean(y_label*(y_pred[:, 0]-logr_pos)**2)
            loss += loss_lp
        if beta2 > 0:
            loss_ln = beta2 * \
                tf.math.reduce_mean(y_label*(y_pred[:, 1]+logr_neg)**2)
            loss += loss_ln

        # positive classified examples should be in the negetive kernel and positive unit sphere
        if beta3 > 0:
            loss_pp = beta3*tf.math.reduce_mean(pos*(y_pred[:, 0]-logr_pos)**2)
            loss += loss_pp
        if beta4 > 0:
            loss_pn = beta4*tf.math.reduce_mean(pos*(y_pred[:, 1]+logr_neg)**2)
            loss += loss_pn

        # Negative classified examples should be in the positive kernel and negative unit sphere
        if beta5 > 0:
            loss_np = beta5*tf.math.reduce_mean(neg*(y_pred[:, 0]+logr_neg)**2)
            loss += loss_np

        if beta6 > 0:
            loss_nn = beta6*tf.math.reduce_mean(neg*(y_pred[:, 1]-logr_pos)**2)
            loss += loss_nn

        # Mean difference of not labeled data should be around 0
        if beta7 > 0:
            diff = beta7*(tf.math.reduce_mean((1.0-y_label) *
                                              y_pred[:, 0]) - tf.math.reduce_mean((1.0-y_label)*y_pred[:, 1]))**2
            loss += diff

        # Mean difference of all data should be around 0
        if beta7all > 0:
            diff = beta7all * \
                (tf.math.reduce_mean(y_pred[:, 0]) -
                 tf.math.reduce_mean(y_pred[:, 1]))**2
            loss += diff

        # Classification loss on classified examples
        if beta8 > 0:
            sel = pos_full + neg_full
            x = y_pred[:, 0] - y_pred[:, 1]
            bce_loss = beta8*bce(pos_full, x, tf.expand_dims(sel, -1))
            loss += bce_loss
        # tf.print([loss_lp, loss_ln, loss_pp, loss_pn,
        #           loss_np, loss_nn, diff, bce_loss])
        return loss

    def pul_loss_unbiased(y_true, y_pred):
        y_label = tf.cast(y_true[:, 0], dtype=tf.float32)

        pos_bool = (y_pred[:, 0]-y_pred[:, 1]) > leps
        neg_bool = (y_pred[:, 1]-y_pred[:, 0]) > leps

        pos_full = tf.cast(pos_bool, dtype=tf.float32)
        neg_full = tf.cast(neg_bool, dtype=tf.float32)

        pos = pos_full * (1.0-y_label)
        neg = neg_full * (1.0-y_label)

        loss_lp = tf.math.reduce_mean(y_label*(y_pred[:, 0]-logr_pos)**2)
        loss_ln = tf.math.reduce_mean(y_label*(y_pred[:, 1]+logr_neg)**2)

        loss_pp = tf.math.reduce_mean(pos*(y_pred[:, 0]-logr_pos)**2)
        loss_pn = tf.math.reduce_mean(pos*(y_pred[:, 1]+logr_neg)**2)

        loss_np = tf.math.reduce_mean(neg*(y_pred[:, 0]+logr_neg)**2)
        loss_nn = tf.math.reduce_mean(neg*(y_pred[:, 1]-logr_pos)**2)

        return loss_lp + loss_ln + loss_pp + loss_pn + loss_np + loss_nn

    return pul_loss, pul_loss_unbiased


def pos_acc(y_true, y_pred):
    y_label = tf.cast(y_true[:, 0], dtype=tf.float32)
    pos = tf.cast(y_pred[:, 0] > y_pred[:, 1], dtype=tf.float32)
    n_pos = tf.reduce_sum(y_label)
    if n_pos == 0:
        return 1.0
    return tf.reduce_sum(pos*y_label)/n_pos


def label_acc(y_true, y_pred):
    y_label = tf.cast(y_true[:, 1], dtype=tf.float32)
    pred_label = tf.cast(y_pred[:, 0] > y_pred[:, 1], dtype=tf.float32)
    pred_check = tf.cast(pred_label == y_label, dtype=tf.float32)
    return tf.reduce_mean(pred_check)


def pos_ratio(y_true, y_pred):
    y_label = tf.cast(y_true[:, 0], dtype=tf.float32)
    pos = tf.cast((y_pred[:, 0] - y_pred[:, 1]) > 0, dtype=tf.float32)
    pos = pos * (1.0-y_label)
    n_pos = tf.reduce_sum(pos)
    n = tf.reduce_sum(1.0-y_label)
    return n_pos/n


def pos_correct(y_true, y_pred):
    npos = tf.reduce_sum(y_true[:, 2])
    ncorrect = tf.reduce_sum(y_true[:, 2]*y_true[:, 1])
    if npos == 0:
        return 1.0
    return ncorrect/npos


def num_add_pos(y_true, y_pred):
    return tf.reduce_sum(y_true[:, 2])


class RegularizerSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, alpha1, up=1.2, down=0.8, damp=0.8, alpha_min=0.5, alpha_max=10.):
        super().__init__()
        self.alpha1 = alpha1
        self.up = up
        self.down = down
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.s = 0
        self.damp = damp

    def on_epoch_end(self, epoch, logs=None):
        if self.alpha1 > self.alpha_max:
            self.alpha1.assign(self.alpha_max)
        if self.alpha1 < self.alpha_min:
            self.alpha1.assign(self.alpha_min)
        if logs["pos_acc"] >= 0.9999 and self.alpha1 < self.alpha_max:
            self.alpha1.assign(self.alpha1*self.up)
            if self.s == -1:
                self.up = (self.up)**self.damp
            self.s = 1
        elif logs["pos_acc"] < 0.99 and self.alpha1 > self.alpha_min:
            self.alpha1.assign(self.alpha1*self.down)
            if self.s == 1:
                self.down = (self.down)**self.damp
            self.s = -1


class AddingPositiveLabelsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, X_train, additional_ipos, ninds=10, nshuffle=0, start_epoch=1, addperiod=10):
        super().__init__()
        self.additional_ipos = additional_ipos
        self.X_train = X_train
        self.model = model
        self.ninds = ninds
        self.nadd = ninds
        self.nshuffle = np.max([nshuffle,ninds])
        self.start_epoch = start_epoch
        self.addperiod = addperiod
        self.max_inds = 200

    def on_epoch_end(self, epoch, logs=None):
        repoch = epoch - self.start_epoch 
        if self.ninds > 0 and self.nshuffle >= self.ninds and repoch > 0:
            self.additional_ipos.clear()
            # if logs["pos_acc"] >= 0.9 and logs["pos_ratio"] < 0.9 and logs["pos_ratio"] > 0.1:
            preds = self.model(self.X_train)
            diff = preds[:, 1]-preds[:, 0]
            sort_order = tf.argsort(diff)

            if repoch % self.addperiod == 0:
                self.nadd += self.ninds
                self.nshuffle += self.ninds
                self.nadd = np.min([self.max_inds,self.nadd])
                self.nshuffle = np.min([self.max_inds,self.nshuffle])

            inds = np.random.choice(range(self.nshuffle),
                                    self.nadd, replace=False)

            for i in inds:
                self.additional_ipos.append(sort_order[i].numpy())


class MPOPUL(tf.keras.Model):
    def __init__(self, D, d=2, S=2, repeat=1, stddev=0.5, alpha1=2.0, alpha2=2.0, alpha3=2., basis=["cos"], dropout=0):
        super(MPOPUL, self).__init__()
        self.D = D
        self.S = S
        self.d = d
        self.basis = basis
        self.repeat = repeat
        self.alpha1 = tf.Variable(initial_value=alpha1, trainable=False)
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.dropout = dropout
        self.embedding = FourierEmbedding(
            repeat=self.repeat, d=self.d, basis=basis)
        self.mpop = Projector(D=D, d=d, S=S, stddev=stddev, dropout=dropout)
        self.mpon = Projector(D=D, d=d, S=S, stddev=stddev, dropout=dropout)
        self.local_sampler = Local_sampler(basis, d=d)

        self.config = {
            "D": D,
            "d": d,
            "S": S,
            "repeat": repeat,
            "stddev": stddev,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "alpha3": alpha3,
            "dropout": dropout,
            "basis": basis,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def call(self, inputs, training):
        x = self.embedding(inputs)
        xp = self.mpop(x, False)
        xn = self.mpon(x, False)
        x0 = tf.stack([xp, xn], axis=1)

        if training:
            if self.dropout > 0. and self.alpha3 > 0:
                xp = self.mpop(x, training)
                xn = self.mpon(x, training)
                x1 = tf.stack([xp, xn], axis=1)
                out_list = [x0, x1]
                self.add_loss(self.alpha3*consistency_loss(out_list))

            lnp = self.mpop.log_norm()
            lnn = self.mpon.log_norm()
            self.add_loss(self.alpha1*tf.abs(lnp))
            # self.add_loss(self.alpha1*tf.abs(lnn))
            self.add_loss(self.alpha2*tf.math.maximum(1.0,
                                                      self.alpha1)*tf.abs(lnn-lnp))
        return x0


class MPOPUL_N(tf.keras.Model):
    def __init__(self, D, d=2, S=2, repeat=1, stddev=0.5, alpha1=2.0, alpha2=2.0, alpha3=2.0, basis=["cos"], nmodels=2):
        super(MPOPUL_N, self).__init__()
        self.D = D
        self.S = S
        self.d = d
        self.basis = basis
        self.repeat = repeat
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.nmodels = nmodels
        self.models = [MPOPUL(D=D, d=d, S=S, stddev=0.5,
                              repeat=repeat, alpha1=alpha1, alpha2=alpha2, basis=basis) for i in range(nmodels)]

        self.config = {
            "D": D,
            "d": d,
            "S": S,
            "repeat": repeat,
            "stddev": stddev,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "basis": basis,
            "nmodels": nmodels,
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def call(self, inputs):
        if self.nmodels == 1:
            return self.models[0](inputs)
        out_list = [model(inputs) for model in self.models]
        self.add_loss(self.alpha3*consistency_loss(out_list))

        out = tf.stack(out_list, axis=2)
        out = tf.reduce_mean(out, axis=2)
        return out


def consistency_loss(out_list):
    n = len(out_list)
    if n == 1:
        return 0
    c = 0
    for i in range(n):
        out1 = out_list[i]
        for j in range(i):
            out2 = out_list[j]
            c += tf.math.reduce_mean((out1-out2)**2)
    return c


def prepare_dataset_gen(x, y, digits=[1, 3], crop=24, Np=100, p=0.5, pdata=0.1, bs=256, augment=True, angle=0.02, scale=0.05, ninds=10, posreal=0.5, additional_ipos=[], istart=0):
    # Select the class which is not an Anomaly
    inds = [k for k in range(len(y)) if y[k] in digits or digits[1] < 0]
    X = x[inds]
    Y = np.array(y[inds])
    N = len(inds)

    imsize = [crop, crop]

    X = tf.expand_dims(X, -1)
    X = np.array(tf.image.resize(X, imsize))

    inds = np.arange(N)
    pos = inds[Y == digits[0]]
    if digits[1] < 0:
        neg = inds[Y != digits[0]]
    else:
        neg = inds[Y == digits[1]]

    nump0 = len(pos)
    numn0 = len(neg)

    nump = nump0
    numn = int(nump*(1-p)/p)

    neg = np.random.choice(neg, numn, replace=numn0 < numn)

    # if numn<numn0:
    #     nump = len(pos)
    #     numn = int(nump*(1-p)/p)
    #     neg = np.random.choice(neg, np.min([numn, len(neg)]), replace=False)
    # else:
    #     numn = len(neg)
    #     nump = int(numn*(p)/(1-p))
    #     pos = np.random.choice(pos, nump, replace=False)

    inds = np.concatenate([neg, pos])

    X = X[inds]
    labels = Y[inds]

    N = len(labels)

    # pos_inds = list(np.random.choice(
    #     np.arange(N)[labels == digits[0]], Np, replace=False))

    all_pos_inds = np.arange(N)[labels == digits[0]]
    pos_inds = all_pos_inds[istart:Np+istart]

    X = 2*X-1

    Y = np.zeros((N, 3))

    neg_inds = [i for i in range(N) if i not in pos_inds]

    # select assign labels 1 (positive) 0 (anomaly) to all examples // this is used for testing and monitoring...
    Y[labels == digits[0], 1] = 1
    Y[pos_inds, 0] = 1          # select positive examples

    Xpos = X[pos_inds]
    Xneg = X[neg_inds]

    Ypos = Y[pos_inds]
    Yneg = Y[neg_inds]

    ipos = np.arange(len(Ypos))
    ineg = np.arange(len(Yneg))

    npos = len(ipos)
    nneg = len(ineg)

    preal = npos/(npos+nneg)

    psamp = np.min([np.max([preal, pdata]), 0.9])
    nsamp = (nneg/(1-psamp))

    def gen():
        while True:
            r = np.random.rand()
            if r < psamp:
                r = np.random.rand()
                if r < posreal or ninds == 0 or len(additional_ipos) < ninds:
                    i = np.random.choice(ipos, 1, replace=True)[0]
                    yield Xpos[i], Ypos[i]
                else:
                    i = np.random.choice(additional_ipos, 1, replace=True)[0]
                    y_out = Y[i]
                    y_out[0] = 1
                    y_out[2] = 1
                    yield X[i], y_out
            else:
                i = np.random.choice(ineg, 1, replace=True)[0]
                yield Xneg[i], Yneg[i]

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec(shape=(crop, crop, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.float32))).batch(bs)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(
            (-angle, angle)),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            [-scale, scale], [-scale, scale])
    ])

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(
            x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds, nsamp, X, labels


def prepare_dataset(x, y, digits=[1, 3], crop=24, Np=100, p=0.5, shuffle=True):
    # Select the class which is not an Anomaly
    inds = [k for k in range(len(y)) if y[k] in digits or digits[1] < 0]
    X = x[inds]
    Y = np.array(y[inds])
    N = len(inds)

    imsize = [crop, crop]

    X = tf.expand_dims(X, -1)
    X = np.array(tf.image.resize(X, imsize))

    inds = np.arange(N)
    pos = inds[Y == digits[0]]
    if digits[1] < 0:
        neg = inds[Y != digits[0]]
    else:
        neg = inds[Y == digits[1]]

    nump0 = len(pos)
    numn0 = len(neg)

    nump = nump0
    numn = int(nump*(1-p)/p)

    neg = np.random.choice(neg, numn, replace=(numn0 < numn))

    # if p >= 0.5:
    #     nump = len(pos)
    #     numn = int(nump*(1-p)/p)
    #     neg = np.random.choice(neg, np.min([numn, len(neg)]), replace=False)
    # else:
    #     numn = len(neg)
    #     nump = int(numn*(p)/(1-p))
    #     pos = np.random.choice(pos, nump, replace=False)
    inds = np.concatenate([neg, pos])
    X = X[inds]
    labels = Y[inds]

    N = len(labels)

    pos_inds = np.random.choice(
        np.arange(N)[labels == digits[0]], Np, replace=False)

    Y = np.zeros((N, 3))

    # select assign labels 1 (positive) 0 (anomaly) to all examples // this is used for testing and monitoring...
    Y[labels == digits[0], 1] = 1
    Y[pos_inds, 0] = 1          # select positive examples

    inds = np.arange(N)
    if shuffle:
        np.random.shuffle(inds)

    return 2*X[inds]-1, Y[inds]

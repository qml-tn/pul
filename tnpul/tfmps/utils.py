import tensorflow as tf
import numpy as np


def init_once(x, name):
    return tf.compat.v1.get_variable(name, initializer=x, trainable=False)


@tf.function
def train_step(model, loss_object, optimizer, train_loss, train_accuracy,
               inputs, labels, regularizer=None):
    vars = model.trainable_variables
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
        if regularizer is not None:
            for v in vars:
                loss += regularizer(v)

    gradients = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(gradients, vars))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(model, loss_object, test_loss, test_accuracy, inputs, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(inputs, training=False)
    t_loss = loss_object(labels, predictions)
    # one_hot_labels = tf.one_hot(labels, 10)
    # t_loss = tf.keras.losses.MSE(one_hot_labels,predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def basis_f(x, j, basis="sin"):
    pi_half = np.pi/2.0
    if basis == "sin":
        return tf.math.sin((x+1)*pi_half*(j+1))
    elif basis == "cos":
        c = 1
        if j == 0:
            c = 1/np.sqrt(2.)
        return c*tf.math.cos((x+1)*pi_half*(j+1))
    else:
        raise Exception(f"Basis {basis} not implemented.")


def pdf(v, x, basis="sin"):
    pi_half = np.pi/2.0
    d = len(v)
    y = []
    for j in range(d):
        y.append(basis_f(x, j, basis))
    y = tf.stack(y, axis=-1)
    y = tf.einsum("oi,ij,oj->o", y, v, y)
    return y


def sample_from_pdf(v_list, basis_list, nbins=1000):
    x = tf.constant(np.linspace(-1, 1, nbins), dtype=tf.float32)
    y = 0
    for v, basis in zip(v_list, basis_list):
        y += pdf(v, x, basis=basis)
    cpd = tf.cumsum(y)
    cpd = cpd/cpd[-1]
    return x[tf.argmax(tf.cast(cpd > np.random.rand(), tf.int32))]


def Local_sampler(blist, d=2, nbins=1000):
    pi_half = np.pi/2.0

    def local_sampler(rholist):
        x = sample_from_pdf(v_list=rholist, basis_list=blist, nbins=nbins)
        out = []
        for basis in blist:
            emb = []
            for j in range(d):
                emb.append(basis_f(x, j, basis))
            out.append(tf.stack(emb)/tf.math.sqrt(1.0*d))
        return x, out
    return local_sampler

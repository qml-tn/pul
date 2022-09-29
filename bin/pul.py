
import tensorflow as tf
import argparse
import os
import numpy as np
import json
import logging
import wandb
from wandb.keras import WandbCallback

from qmltn.tfmps.pul import pos_acc, label_acc, MPOPUL, PULloss, PULloss2, pos_correct, num_add_pos, RegularizerSchedulerCallback, AddingPositiveLabelsCallback, pos_ratio, prepare_dataset, prepare_dataset_gen
from qmltn import __version__


class ModifiedWandbCallback(WandbCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}):
        logs.update({'lr': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)


class PUL_Trainer():
    def __init__(self):
        parser = self.arg_parser()

        parse_args = self.arg_parser().parse_args()

        # Determining the model name for saving the model

        kwargs = parse_args.__dict__
        self.kwargs = kwargs

        # Verbosity
        verbose = kwargs["verbose"]
        self.verbose = logging.WARNING
        if verbose == 1:
            self.verbose = logging.INFO
        elif verbose == 2:
            self.verbose = logging.DEBUG
        logging.basicConfig(format='%(asctime)s | %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=self.verbose)

        logging.info(f"Initializing trainer v {__version__}")

        self.cuda = kwargs["cuda"]

        self.dataset = kwargs["dataset"]
        self.savedir = kwargs["savedir"]
        self.datadir = kwargs["datadir"]

        self.positive_class = kwargs["positive_class"]
        self.negative_class = kwargs["negative_class"]
        self.np = kwargs["np"]
        self.p = kwargs["p"]
        self.crop = kwargs["crop"]

        self.wandb = kwargs["wandb"]
        self.wandb_project = "PUL"
        if kwargs["entity"] == "":
            self.entity = None
        else:
            self.entity = kwargs["entity"]
        self.wandb_name = None
        self.wandb_offline = kwargs["wandb_offline"]

        self.repeat = kwargs["repeat"]
        self.basis = kwargs["basis"]
        self.d = kwargs["d"]
        self.D = kwargs["D"]
        self.S = kwargs["S"]

        self.bs = kwargs["bs"]
        self.ntrain = kwargs["ntrain"]
        self.ntest = kwargs["ntest"]
        self.nepoch = kwargs["nepoch"]

        self.seed = kwargs["seed"]

        self.lr = kwargs["lr"]
        self.gamma = kwargs["gamma"]
        self.step = kwargs["step"]
        self.stop_patience = kwargs["stop_patience"]

        self.alpha1 = kwargs["alpha1"]
        self.alpha2 = kwargs["alpha2"]
        self.alpha3 = kwargs["alpha3"]
        self.up = kwargs["up"]
        self.down = kwargs["down"]
        self.alpha_min = kwargs["alpha_min"]
        self.alpha_max = kwargs["alpha_max"]

        self.beta1 = kwargs["beta1"]
        self.beta2 = kwargs["beta2"]
        self.beta3 = kwargs["beta3"]
        self.beta4 = kwargs["beta4"]
        self.beta5 = kwargs["beta5"]
        self.beta6 = kwargs["beta6"]
        self.beta7 = kwargs["beta7"]
        self.beta8 = kwargs["beta8"]
        self.leps = kwargs["leps"]
        self.logr_pos = kwargs["logr_pos"]
        self.logr_neg = kwargs["logr_neg"]
        self.labp = kwargs["labp"]
        self.weighted = kwargs["weighted"]

        self.augment = kwargs['augment']
        self.angle = kwargs['angle']
        self.scale = kwargs['scale']
        self.dropout = kwargs["dropout"]

        self.ninds = kwargs["ninds"]
        self.nshuffle = kwargs["nshuffle"]
        self.posreal = kwargs["posreal"]
        self.istart = kwargs["istart"]
        self.estart = kwargs["estart"]
        self.addperiod = kwargs["addperiod"]

        loss, unbiased_loss = PULloss(beta1=self.beta1, beta2=self.beta2, beta3=self.beta3, beta4=self.beta4, beta5=self.beta5,
                                      beta6=self.beta6, beta7=self.beta7, beta8=self.beta8, leps=self.leps, logr_pos=self.logr_pos, 
                                      logr_neg=self.logr_neg, weighted=self.weighted)

        self.model = MPOPUL(D=self.D, d=self.d, S=self.S, stddev=0.5,
                            repeat=self.repeat, alpha1=self.alpha1, alpha2=self.alpha2, alpha3=self.alpha3, dropout=self.dropout, basis=self.basis)

        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

        self.additional_ipos = []

        # parameter monitoring
        def model_alpha1(y_true, y_pred):
            return self.model.alpha1

        # norm logging
        def log_norm_positive(y_true, y_pred):
            return self.model.mpop.log_norm()

        def log_norm_negative(y_true, y_pred):
            return self.model.mpon.log_norm()

        self.model.compile(optimizer=self.optimizer,
                           loss=loss, metrics=[pos_acc, label_acc, unbiased_loss, pos_ratio, model_alpha1, log_norm_positive, log_norm_negative, pos_correct, num_add_pos])

        # For now we test only with the MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train/255.0
        x_test = x_test/255.0

        digits = [self.positive_class, self.negative_class]

        # Preparing the train and test datasets
        # self.X_train, self.Y_train = prepare_dataset(
        #     x_train, y_train, digits=digits, crop=self.crop, Np=self.np, p=self.p)

        self.ds_train, self.nsamp, self.X_train, self.labels = prepare_dataset_gen(
            x_train, y_train, digits=digits, crop=self.crop, Np=self.np, p=self.p, bs=self.bs,
            pdata=self.labp, augment=self.augment, angle=self.angle, scale=self.scale, 
            ninds=self.ninds, posreal=self.posreal, additional_ipos=self.additional_ipos, istart=self.istart)

        self.X_test, self.Y_test = prepare_dataset(
            x_test, y_test, digits=digits, crop=self.crop, Np=self.np, p=self.p)

    def train(self):
        callbacks = []
        if self.wandb:
            wandb.init(
                entity=self.entity, project=f"{self.dataset}_{self.wandb_project}",
                tags=["tf"], name=self.wandb_name, config=self.kwargs)
            callbacks.append(ModifiedWandbCallback(save_model=False))

        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='pul_loss_unbiased', factor=self.gamma, patience=self.step, verbose=0,
            mode='auto', min_delta=1e-5, cooldown=0, min_lr=1e-6))

        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='pul_loss_unbiased', min_delta=0, patience=self.stop_patience, verbose=0,
            mode='auto', baseline=None, restore_best_weights=False
        ))

        callbacks.append(RegularizerSchedulerCallback(self.model.alpha1, up=self.up,
                                                      down=self.down, alpha_min=self.alpha_min, alpha_max=self.alpha_max))

        callbacks.append(AddingPositiveLabelsCallback(
            self.model, self.X_train, self.additional_ipos, ninds=self.ninds, start_epoch=self.estart,addperiod=self.addperiod))

        self.model.fit(self.ds_train, batch_size=self.bs,
                       epochs=self.nepoch, steps_per_epoch=int(
                           self.nsamp/self.bs)+1, callbacks=callbacks, verbose=self.kwargs["verbose"],
                       validation_data=(self.X_test, self.Y_test))

        # saving the final model
        # TODO: save only the best model. Smallest loss and pos_acc=1 and and constant pos_ration in the range 0.1<pos_ratio >0.9
        if self.wandb:
            # self.model.save(os.path.join(
            #     wandb.run.dir, "model"),  save_format="tf")
            self.model.save_weights(os.path.join(
                wandb.run.dir, "weights.h5"))
            # self.model.save_weights(os.path.join(
            #     wandb.run.dir, "ckpt"))

            wandb.finish()
        # self.model.evaluate(self.X_test, self.Y_test, batch_size=self.bs,
        #                     epochs=self.nepoch, callbacks=callbacks, verbose=self.kwargs["verbose"])

    # def evaluate(self):
    #     eval = self.model.evaluate(self.test_ds)
    #     # Saving evaluation results
    #     data = json.dumps(eval)
    #     f = open(os.path.join(self.savedir, "evaluation.json"), "w")
    #     f.write(data)
    #     f.close()

    def save_config(self):
        data = json.dumps(self.kwargs)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
        f = open(os.path.join(self.savedir, "conf.json"), "w")
        f.write(data)
        f.close()

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--verbose',
                            default=0,
                            type=int,
                            help='Determines the logger output.')

        parser.add_argument('--cuda',
                            default=1,
                            type=int,
                            help='Use cuda GPU. 1 (default)')

        parser.add_argument('--dataset',
                            default="MNIST",
                            type=str,
                            help='The name of the dataset')
        parser.add_argument('--savedir',
                            default="../data/tdvp",
                            type=str,
                            help='A full path to where the files should be stored.')
        parser.add_argument('--datadir',
                            default="../dataset/",
                            type=str,
                            help='A full path to where the files should be stored.')

        parser.add_argument('--positive_class',
                            default=1,
                            type=int,
                            help='Selection of the index of the positive and negative class. 1 (default)')
        parser.add_argument('--negative_class',
                            default=3,
                            type=int,
                            help='Selection of the index of the negative and negative class. 3 (default)')
        parser.add_argument('--np',
                            default=100,
                            type=int,
                            help='Number of labeled positive examples. 100 (default)')
        parser.add_argument('--istart',
                            default=0,
                            type=int,
                            help='First index in the list of positive labeled examples. 0 (default)')
        parser.add_argument('--ninds',
                            default=30,
                            type=int,
                            help='Number of additional labeled positive examples added at the end of each epoch. 30 (default)')
        parser.add_argument('--nshuffle',
                            default=0,
                            type=int,
                            help='Number of additional examples considered to be added to labeled positive examples at the end of each epoch. If nshuffle is smaller as ninds we take best ninds examples. 0 (default)')
        parser.add_argument('--estart',
                            default=1,
                            type=int,
                            help='Epoch after which we should add new positive labeled examples. 1 (default)')
        parser.add_argument('--addperiod',
                            default=1,
                            type=int,
                            help='Period with which we increase the number of new positive labeled examples by ninds. 1 (default)')
        parser.add_argument('--p',
                            default=0.5,
                            type=float,
                            help='Ratio of unlabeled positive examples. 0.5 (default)')
        parser.add_argument('--crop',
                            default=25,
                            type=int,
                            help='Image crop size. 25 (default)')
        parser.add_argument('--labp',
                            default=0.3,
                            type=float,
                            help='Ratio of labeled examples in each bach. 0.3 (default)')
        parser.add_argument('--posreal',
                            default=0.5,
                            type=float,
                            help='Ratio of real positive examples (relative to all positive examples). Used only if new examples are added on the fly. 0.5 (default)')

        parser.add_argument('--wandb',
                            default=1,
                            type=int,
                            help='Enabling wandb monitoring. 1 (default)')
        parser.add_argument('--wandb_offline',
                            default=0,
                            type=int,
                            help='Used only if wandb == 1. If enabled the logging is done offline. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--entity',
                            default="",
                            type=str,
                            help='WANDB entity where to send the runs data. Default is empty meaning the runs will be sent to default entity. Only used if wandb is enabled.')

        parser.add_argument('--repeat',
                            default=1,
                            type=int,
                            help='Repetition of the input. 1 (default)')
        parser.add_argument('--basis',
                            nargs='+',
                            default=["cos"],
                            type=str,
                            help='A list ob basis used. The length of the list should be the same as "repeat". cos(default), sin.')
        parser.add_argument('--d',
                            default=20,
                            type=int,
                            help='Size of the feature space/number of modes.')
        parser.add_argument('--D',
                            default=30,
                            type=int,
                            help='Bond dimension for the mps.')
        parser.add_argument('--S',
                            default=3,
                            type=int,
                            help='Output skip-step in the MPO.')

        parser.add_argument('--bs',
                            default=500,
                            type=int,
                            help='Batch size. 500 (default)')
        parser.add_argument('--nepoch',
                            type=int,
                            default=1000,
                            help='Number of epochs. 1000 (default)')
        parser.add_argument('--ntrain',
                            default=60000,
                            type=int,
                            help='Number of training examples. Defaults to MNIST size.')
        parser.add_argument('--ntest',
                            default=10000,
                            type=int,
                            help='Number of test examples. Defaults to MNIST size.')

        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help='Seed for the optimization and initialization. 0 (default)')

        parser.add_argument('--lr',
                            default=0.0002,
                            type=float,
                            help='Learning rate. 0.0002 (default)')
        parser.add_argument('--gamma',
                            default=0.5,
                            type=float,
                            help='Learning rate decay: lr = lr * gamma. 0.5 (default)')
        parser.add_argument('--step',
                            default=50,
                            type=int,
                            help='Patience for the learning rate decay in the number of epochs. 10 (default)')
        parser.add_argument('--stop_patience',
                            default=700,
                            type=int,
                            help='Early stopping patience in epochs. 50 (default)')

        parser.add_argument('--alpha1',
                            default=1.0,
                            type=float,
                            help='Loss parameter (positive normalisation). 1.0 (default)')
        parser.add_argument('--alpha2',
                            default=1.0,
                            type=float,
                            help='Loss parameter (negative-positive normalisation ratio). 1.0 (default)')
        parser.add_argument('--alpha3',
                            default=0,
                            type=float,
                            help='Loss parameter (consistency loss). 1.0 (default)')
        parser.add_argument('--dropout',
                            default=0.,
                            type=float,
                            help='Dropout probability. 0. (default)')
        parser.add_argument('--up',
                            default=1.1,
                            type=float,
                            help='Exponential increase of alpha1. 1.1 (default)')
        parser.add_argument('--down',
                            default=0.9,
                            type=float,
                            help='Exponential decrease of alpha1. 0.9 (default)')
        parser.add_argument('--alpha_min',
                            default=0.1,
                            type=float,
                            help='Minimum of alpha1. 0.1 (default)')
        parser.add_argument('--alpha_max',
                            default=300,
                            type=float,
                            help='Maximum of alpha1. 300 (default)')

        parser.add_argument('--beta1',
                            default=1.0,
                            type=float,
                            help='Positive labeled loss strength. 1.0 (default)')
        parser.add_argument('--beta2',
                            default=1.0,
                            type=float,
                            help='Negative labeled loss strength. 1.0 (default)')
        parser.add_argument('--beta3',
                            default=1.0,
                            type=float,
                            help='Positive positive examples loss strength. 1.0 (default)')
        parser.add_argument('--beta4',
                            default=1.0,
                            type=float,
                            help='Negative positive examples loss strength. 1.0 (default)')
        parser.add_argument('--beta5',
                            default=1.0,
                            type=float,
                            help='Positive negative examples loss strength. 1.0 (default)')
        parser.add_argument('--beta6',
                            default=1.0,
                            type=float,
                            help='Negative negative examples loss strength. 1.0 (default)')
        parser.add_argument('--beta7',
                            default=1.0,
                            type=float,
                            help='Square of the mean difference between the negative and positive logits. 1.0 (default)')
        parser.add_argument('--beta8',
                            default=1.0,
                            type=float,
                            help='Binary cross entropy losss. 1.0 (default)')

        parser.add_argument('--leps',
                            default=20,
                            type=float,
                            help='LogRatio selection minimum between positive and negative probability. 20 (default)')
        parser.add_argument('--logr_pos',
                            default=5,
                            type=float,
                            help='Target value of the log probability for positive examples. 5 (default)')
        parser.add_argument('--logr_neg',
                            default=50,
                            type=float,
                            help='Target value of the log probability for negative examples. 50 (default)')
        parser.add_argument('--weighted',
                            default=0,
                            type=int,
                            help='If enabled we use soft sigmoid labels to define probable positive and negative data. 0 (default).')
        parser.add_argument('--augment',
                            default=1,
                            type=int,
                            help='If enabled image augmentation is applied to all examples. 1 (default)')
        parser.add_argument('--angle',
                            default=0.02,
                            type=float,
                            help='Augmentation rotation angle ratio of 2pi. 0.02 (default)')
        parser.add_argument('--scale',
                            default=0.05,
                            type=float,
                            help='Zoom scale percentate. 0.05 (default)')
        return parser


if __name__ == "__main__":
    logging.info(f"Using qmltn version {__version__}")
    logging.info("Initializing the TDVP model")
    pul = PUL_Trainer()
    logging.info("Saving the config")
    pul.save_config()
    logging.info("Training the model")
    pul.train()

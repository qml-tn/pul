from tnpul.torchmps.tuner.tuner import Tuner
from kerastuner import HyperParameters
import argparse
import logging
from tnpul import __version__


class TunerCLI(Tuner):
    def __init__(self, init_loaders_all_folds):
        hp = HyperParameters()
        self.parse_args = self.arg_parser().parse_args()
        kwargs = vars(self.parse_args)

        # Verbosity
        verbose = kwargs["verbose"]
        self.verbose = logging.WARNING
        if verbose == 1:
            self.verbose = logging.INFO
        elif verbose == 2:
            self.verbose = logging.DEBUG
        logging.basicConfig(format='%(asctime)s | %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=self.verbose)

        logging.info("Initializing tuner version {__version__}")

        # Defining training parameters hyperparameters to tune
        bs = hp.Int('bs', 100, 500, step=100, default=100)
        lr = hp.Float('lr', 1e-7, 1e-2,
                      sampling='log', default=1e-4)
        l2 = hp.Float('l2', 1e-7, 1e-2,
                      sampling='log', default=1e-3)
        optimizer = hp.Choice("optimizer",
                              ["adam", "adadelta", "adamw", "adamax", "rmsprop", "rprop", "sgd"])
        step = hp.Int('step', 5, 20, step=5, default=10)
        gamma = hp.Float('gamma', 0.1, 0.9, step=0.1, default=0.5)
        # These parameters do not have any effect since nepochs is usually <=30 due to time constraints
        step = hp.Fixed('step', 50)
        gamma = hp.Fixed('gamma', 0.5)

        embedding = hp.Fixed('embedding', 'linear')
        if kwargs["disable_emb_tun"]:
            logging.info("Disabling embedding tuning")
            emb_ord = hp.Fixed('embedding_order', 1)
        else:
            emb_ord = hp.Choice("embedding_order", [1, 2, 3])

        if kwargs["crop"] <= 0:
            cropMax = 28
            if kwargs["dataset"] in ["CIFAR10", "CIFAR100"]:
                cropMax = 32
            crop = hp.Int("crop", 22, cropMax, default=24)
        else:
            crop = hp.Fixed('crop', kwargs['crop'])
            logging.info(f"Using predetermined crop: {kwargs['crop']}")

        aug_phi = hp.Float('aug_phi', 1e-7, 1e-1,
                           sampling='log', default=1e-3)
        self.hp = hp

        super(TunerCLI, self).__init__(
            init_loaders_all_folds=init_loaders_all_folds, **kwargs)

    def get_hyperparameters(self):
        return self.hp

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--dataset',
        #                     default="TEST",
        #                     type=str,
        #                     help='The name of the dataset.')
        # parser.add_argument('--savedir',
        #                     default="../data/test",
        #                     type=str,
        #                     help='A full path to where the files should be stored.')
        parser.add_argument('--prefix',
                            default="",
                            type=str,
                            help='Prefix for saved models. Empty string by default.')
        parser.add_argument('--nfolds',
                            default=5,
                            type=int,
                            help='Number of folds for crossvalidation: 5 (default).')
        parser.add_argument('--wandb',
                            default=0,
                            type=int,
                            help='If enabled the results will be send to wandb. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--wandb_offline',
                            default=0,
                            type=int,
                            help='Used only if wandb == 1. If enabled the logging is done offline. 0 (disabled-default), 1 (enabled).')
        parser.add_argument('--entity',
                            default="",
                            type=str,
                            help='WANDB entity where to send the runs data. Default is empty meaning the runs will be sent to default entity. Only used if wandb is enabled.')
        parser.add_argument('--oracle',
                            default="bayes",
                            type=str,
                            help='KerasTuner oracle to use: bayes (default), random, hyperband')
        parser.add_argument('--max_training_hours',
                            default=44,
                            type=float,
                            help='Maximal training duration in hours. 44 (default), -1 => no simulation time limit.')
        parser.add_argument('--crop',
                            default=-1,
                            type=int,
                            help='If crop>0, we disable the crop parameter search and fix the value. -1 (default) crop search is enabled.')
        parser.add_argument('--continue_training',
                            default=0,
                            type=int,
                            help='If enabled a new job will be submitted to the queue after the job is killed before completion: 0 (default), 1')
        parser.add_argument('--checkpoint',
                            default="",
                            type=str,
                            help='A path to the model for from which we start training. If not specified we use a standard random initialization. If a model already exists, the checkpoint is ignored.')
        parser.add_argument('--stop_patience',
                            default=8,
                            type=int,
                            help='Early stopping patience in epochs. 50 (default)')
        parser.add_argument('--reset_early_stopping',
                            default=0,
                            type=int,
                            help='If enabled resets the early stopping difference to 0.')
        parser.add_argument('--datadir',
                            default="../dataset/",
                            type=str,
                            help='A full path to where the files should be stored.')
        parser.add_argument('--D',
                            default=10,
                            type=int,
                            help='Bond dimension for the mps.')
        parser.add_argument('--d',
                            default=2,
                            type=int,
                            help='Local Hilbert space dimension.')
        # parser.add_argument('--bs',
        #                     default=500,
        #                     type=int,
        #                     help='Batch size.')
        parser.add_argument('--ntrain',
                            default=60000,
                            type=int,
                            help='Number of training examples. Defaults to MNIST size.')
        parser.add_argument('--ntest',
                            default=10000,
                            type=int,
                            help='Number of test examples. Defaults to MNIST size.')
        parser.add_argument('--train_ratio',
                            default=1.0,
                            type=int,
                            help='Ratio of the training set examples to be used in training.')
        parser.add_argument('--profile',
                            default=0,
                            type=int,
                            help='If enabled the model profiling will be performed before the start of the training. 0 (disabled-default), 1 (enabled).')
        # parser.add_argument('--lr',
        #                     default=0.0002,
        #                     type=float,
        #                     help='Learning rate.')
        # parser.add_argument('--l2',
        #                     default=0.0,
        #                     type=float,
        #                     help='L2 regularization of the MPS parameters. Default is 0.')
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help='Seed for training and dataset generation. 42 (default)')
        parser.add_argument('--nepoch',
                            type=int,
                            default=100,
                            help='Number of epoch.')
        # parser.add_argument('--step',
        #                     default=50,
        #                     type=int,
        #                     help='Step size for the learning rate decay in the number of epochs.')
        # parser.add_argument('--gamma',
        #                     default=0.5,
        #                     type=float,
        #                     help='Learning rate decay: lr = lr * gamma.')
        parser.add_argument('--permute',
                            action='store_true',
                            help='Enable permutation of the input.')
        parser.add_argument('--spiral',
                            action='store_true',
                            help='Spiral order of mps sites in the image.')
        parser.add_argument('--periodic',
                            action='store_true',
                            help='Enable periodic boundary conditions.')
        parser.add_argument('--ti',
                            action='store_true',
                            help='Translationary invariant MPS.')
        parser.add_argument('--cuda',
                            action='store_true',
                            help='Use cuda GPU.')
        parser.add_argument('--nclass',
                            default=10,
                            type=int,
                            help='Number of classes.')
        parser.add_argument('--verbose',
                            default=0,
                            type=int,
                            help='Determines the logger output.')
        parser.add_argument('--monitoring',
                            action='store_true',
                            help='If enabled the model will be evaluated on the test set after each epoch.')
        parser.add_argument('--savemodel',
                            action='store_true',
                            help='If enabled the model will be saved after each epoch.')
        parser.add_argument('--disable_color_jitter',
                            action='store_true',
                            help='Disabled color jitter transformation for tuning.')
        parser.add_argument('--disable_sharpness',
                            action='store_true',
                            help='Disabled sharpness transformation for tuning.')
        parser.add_argument('--disable_blur',
                            action='store_true',
                            help='Disabled blur transformation for tuning.')
        parser.add_argument('--disable_hflip',
                            action='store_true',
                            help='Disabled horizontal filp transformation for tuning.')
        parser.add_argument('--disable_affine',
                            action='store_true',
                            help='Disabled affine transformation for tuning.')
        parser.add_argument('--disable_perspective',
                            action='store_true',
                            help='Disabled perspective transformation for tuning.')
        parser.add_argument('--disable_elastic',
                            action='store_true',
                            help='Disabled elastic transformation for tuning.')
        parser.add_argument('--disable_erasing',
                            action='store_true',
                            help='Disabled perspective transformation for tuning.')
        parser.add_argument('--disable_emb_tun',
                            action='store_true',
                            help='Disabled tuning of the embedding order. Set to 1.')

        return parser

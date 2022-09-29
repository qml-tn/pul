import kerastuner as kt
import argparse
import numpy as np
import logging
from abc import abstractmethod
import os

from tnpul.utils.killer import GracefulKiller
from tnpul.torchmps.torchmps import build_mps_model
from tnpul.torchmps.trainer.trainer import Trainer


class ImageTrainer(Trainer):
    def __init__(self, init_loaders, *args, **kwargs):
        self.init_loaders_ = init_loaders
        self.killer = GracefulKiller(disable=True)
        self.fold = kwargs["fold"]
        super(ImageTrainer, self).__init__(*args, **kwargs, killer=self.killer)

    def init_loaders(self, *args, **kwargs):
        loaders, num_batches = self.init_loaders_(*args, **kwargs)
        self.loaders = loaders
        self.num_batches = num_batches

    def get_wandb_project(self, *args, **kwargs):
        return "MPS"

    def get_feature_dim(self, *args, **kwargs):
        # The standard embedding has a local dimension nchan+1.
        feature_dim = 2
        if kwargs["dataset"] in ["CIFAR10", "CIFAR100"]:
            # For color datasets we use a different embedding with local hilbert space of dimension 4
            feature_dim = 4
        
        # We can add additional higher order features for linear embeddings
        nchan = feature_dim - 1
        if kwargs["embedding"] == "linear":
            emb_ord = kwargs["embedding_order"]
            if emb_ord == 2:
                feature_dim += nchan**2
            elif emb_ord == 3:
                feature_dim += nchan**2 + nchan**3

        print("feature_dim",feature_dim)
        return feature_dim

    def get_input_dim(self, *args, **kwargs):
        return kwargs['crop']**2

    def build_model(self, *args, **kwargs):
        self.model, config = build_mps_model(**self.config)
        self.epoch = config["epoch"]
        self.best_val_acc = config["best_val_acc"]
        self.best_val_acc_epoch = config["best_val_acc_epoch"]
        self.num_model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config["num_model_params"] = self.num_model_params

    def get_model_name(self, *args, **kwargs):
        modelname = f"{self.prefix}D{self.bond_dim}l{self.learn_rate}s{self.seed}_f{self.fold}_{self.nfolds}"
        if self.permute:
            modelname += "p"
        if self.spiral:
            modelname += "s"
        if self.periodic_bc or self.ti:
            modelname += "c"
        else:
            modelname += "o"
        if self.ti:
            modelname += "ti"

        # Saving models into a different folder
        self.savedir = os.path.join(self.savedir, "models")
        return modelname


class TrainerHypermodel(kt.HyperModel):
    def __init__(self, *args, **kwargs):
        super(TrainerHypermodel, self).__init__()
        self.init_loaders_all_folds = kwargs.pop('init_loaders_all_folds')
        self.args = args
        self.kwargs = kwargs

    def build(self, hp, fold=0):
        self.kwargs.update(hp.values)
        cv_loaders, cv_num_batches = self.init_loaders_all_folds(
            hp, *self.args, **self.kwargs)
        assert fold < len(
            cv_loaders), f"Trying to access the fold {fold}, but the number of folds is {len(cv_loaders)}."

        def init_loaders(*args, **kwargs):
            return cv_loaders[fold], cv_num_batches[fold]

        return ImageTrainer(init_loaders, *self.args, fold=fold, **self.kwargs)


class Tuner(kt.engine.base_tuner.BaseTuner):
    def __init__(self, *args, **kwargs):
        # Initializing the tuner parameters

        hp = self.get_hyperparameters()
        kwargs.update(hp.values)

        self.kwargs = kwargs

        self.dataset = kwargs["dataset"]
        self.savedir = kwargs["savedir"]

        self.oracle = kwargs["oracle"]

        self.mpshypermodel = TrainerHypermodel(*args, **kwargs)

        if self.oracle == "bayes":
            logging.info(f"Using BayesianOptimization oracle.")
            oracle = kt.oracles.BayesianOptimization(
                hyperparameters=hp,
                objective=kt.Objective("accuracy", "max"),
                max_trials=1000
            )
        elif self.oracle == "random":
            logging.info(f"Using RandomSearchOracle.")
            oracle = kt.oracles.RandomSearch(
                objective=kt.Objective("accuracy", "max"),
                hyperparameters=hp,
                max_trials=1000
            )
        elif self.oracle == "hyperband":
            logging.info(f"Using HyperBand oracle.")
            oracle = kt.oracles.Hyperband(
                objective=kt.Objective("accuracy", "max"),
                hyperparameters=hp,
                max_epochs=30,
                factor=3,
                hyperband_iterations=3
            )
        else:
            raise Exception(
                f"The oracle {self.oracle} is not supported. Supported oracles are: random, bayes (default), hyperband.")

        super(Tuner, self).__init__(oracle=oracle, hypermodel=self.mpshypermodel,
                                    directory=self.savedir, project_name=self.dataset)

    def run_trial(self, trial):
        hp = trial.hyperparameters
        # We could add hyperparameters also here, i.e. directly before running the trial.
        acc_list = []
        for fold in range(self.kwargs["nfolds"]):
            logging.info(f"###### FOLD {fold}/{self.kwargs['nfolds']} ######")
            model = self.mpshypermodel.build(hp, fold)
            model.train()
            val_acc = model.validate()
            acc_list.append(val_acc)

        acc = np.mean(acc_list)

        logging.info(f"Fold accuracies {acc_list}.\n")
        logging.info(
            f"Accuracy mean  = {acc}, Accuracy std {np.std(acc_list)}.\n")

        # Updating the kerastuner score
        self.oracle.update_trial(trial.trial_id, {"accuracy": acc})

    @abstractmethod
    def get_hyperparameters(self):
        raise NotImplementedError()

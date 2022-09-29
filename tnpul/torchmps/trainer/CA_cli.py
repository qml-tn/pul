import argparse

import os
import torch
import logging
from tqdm import tqdm
import numpy as np
import time
import wandb

from tnpul.torchmps.trainer.base import Trainer
from tnpul.utils.dataset import init_loaders
from tnpul.torchmps.embeddings import image_embedding, linear_encoder
from tnpul.torchmps.tdvp.mps_tdvp import CA_TDVP_V2
from tnpul.torchmps.trainer.cli import TrainerCLI
from tnpul.torchmps.trainer.base import ReduceLROnPlateau

from sklearn.decomposition import PCA


def rule153(x, j, cyclic=False):
    if cyclic:
        x0 = x[..., :j]+1
    else:
        x0 = torch.zeros([x.shape[0], j])

    xr = torch.cat([x[..., j:]+1, x0], axis=1)
    xt = torch.tensor(x)
    out = (xt + xr) % 2
    return out


def rule30(x, cyclic=False):
    if isinstance(x, torch.Tensor):
        if cyclic:
            x0 = x[..., :1]
            xn = x[..., -1:]
        else:
            x0 = torch.zeros([x.shape[0], 1])
            xn = torch.zeros([x.shape[0], 1])
        xr = torch.cat([x[..., 1:], x0], axis=1)
        xl = torch.cat([xn, x[..., :-1]], axis=1)
        xt = x
        out = (xl+xt+xr+xt*xr) % 2
    else:
        if cyclic:
            x0 = torch.tensor(x[..., :1])
            xn = torch.tensor(x[..., -1:])
        else:
            x0 = torch.zeros([x.shape[0], 1])
            xn = torch.zeros([x.shape[0], 1])
        xr = torch.cat([torch.tensor(x[..., 1:]), x0], axis=1)
        xl = torch.cat([xn, torch.tensor(x[..., :-1])], axis=1)
        xt = torch.tensor(x)
        out = (xl+xt+xr+xt*xr) % 2
    return out.float()


def rule30j(x, j=1, cyclic=False):
    for i in range(j):
        x = rule30(x, cyclic=cyclic)
    return x


def average_domain_size(y_true, y_pred):
    n = y_pred.shape[-1]
    x = torch.abs(torch.round(y_pred)-y_true)
    return torch.mean(1+torch.sum(1-torch.abs(x[..., 1:]-x[..., :-1]), axis=-1)/(torch.sum(torch.abs(x[..., 1:]-x[..., :-1]), axis=-1)+1))/n


def rotate(x, t):
    return torch.cat([x[..., t:], x[..., :t]], axis=-1)


def correlations(y_true, y_pred):
    n = y_pred.shape[-1]
    x = 2*torch.abs(torch.round(y_pred)-y_true)-1
    corr = np.zeros(n//2+1)
    for i in range(n//2+1):
        corr[i] = torch.mean(x*rotate(x, i+1))
    return corr, x.detach().numpy()


class CellularAutomataTrainerCLI_V2(TrainerCLI):
    def __init__(self):
        super(CellularAutomataTrainerCLI_V2, self).__init__()

        kwargs = self.config
        self.rule = kwargs['rule']
        self.rule_j = kwargs['rule_j']
        self.n = kwargs['n']
        self.crop = kwargs['crop']
        self.disable_tqdm = kwargs['disable_tqdm']
        self.l1 = kwargs['l1']
        self.l2 = kwargs['l2']
        self.din = kwargs["feature_dim"]
        self.dout = kwargs["dout"]
        self.cyclic = kwargs["cyclic"]

        self.last_checkpoint_epoch = 0
        self.embedding = linear_encoder

        self.torch_one = torch.tensor(
            1.0, dtype=torch.float32, device=self.device)
        self.torch_zero = torch.tensor(
            0.0, dtype=torch.float32, device=self.device)

    def init_loaders(self, *args, **kwargs):
        rule = kwargs['rule']
        rule_j = kwargs['rule_j']
        ntrain = kwargs['ntrain']
        n = kwargs['nmax']
        bs = kwargs['bs']

        if kwargs['dataset_mode'] == "all":
            assert ntrain == 2**n, "ntrain should be equal to 2**n"
            X = np.array([[int(c) for c in bin(i)[2:].zfill(n)]
                          for i in range(2**n)])
        else:
            X = np.round(np.random.rand(ntrain, n))
        tensor_y = self.rule_fun(X)
        # transform to torch tensor if necessary
        tensor_x = torch.tensor(X, dtype=torch.float32)
        # tensor_y = torch.tensor(tensor_y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

        self.train_ds = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=bs, drop_last=True)

        ntest = self.config["ntest"]
        n = 1000
        self.Xtest = torch.tensor(
            np.round(np.random.rand(ntest, n)), dtype=torch.float32)
        self.Ytest = self.rule_fun(self.Xtest)

        if self.cuda:
            self.Xtest = self.Xtest.to(self.device)
            self.Ytest = self.Ytest.to(self.device)

    def get_feature_dim(self, *args, **kwargs):
        return 2

    def get_input_dim(self, *args, **kwargs):
        return kwargs['nmax']

    def init_optimizer(self):
        logging.info(f"Initializing the optimizer.")
        # Set our loss function, optimizer, and scheduler
        self.loss_fun = self.init_loss()
        self.lr_classifier = self.config["lr_class"]
        self.lr_context = self.learn_rate

        context_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if name.endswith("B"):
                classifier_params.append(param)
            else:
                context_params.append(param)

        if self.lr_classifier > 0:
            model_params = [{"params": context_params, "lr": self.lr_context}, {
                "params": classifier_params, "lr": self.lr_classifier, "weight_decay": self.l2_reg}]
        else:
            model_params = self.model.parameters()

        print("classifier params", classifier_params[0].data[0, 0, :, :])

        if self.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(model_params, lr=self.learn_rate,
                                              weight_decay=self.l2_reg)
        elif self.optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(model_params, lr=self.learn_rate,
                                             momentum=0, dampening=0, weight_decay=self.l2_reg, nesterov=False)
        else:
            logging.warning(
                "Specified optimizer not found using the default Adam optimizer.")
            self.optimizer = torch.optim.Adam(model_params, lr=self.learn_rate,
                                              weight_decay=self.l2_reg)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=self.gamma, patience=self.step_size,
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08,
            verbose=self.verbose in [logging.INFO, logging.DEBUG],
            num_bad_epochs=self.epoch-self.best_val_acc_epoch, best_accuracy=self.best_val_acc)

        # We need to update this for graceful killer to have the learning rate even if the trainer has not finished one step.
        self.scheduler._last_lr = [self.learn_rate]

    def rule_fun(self, X):
        cyclic = self.config["cyclic"]
        rule = self.config['rule']
        rule_j = self.config['rule_j']
        if rule == 153:
            Y = rule153(X, rule_j, cyclic)
        elif rule == 30:
            Y = rule30j(X, rule_j, cyclic)
        return Y

    def build_model(self, *args, **kwargs):
        model_path = self.model_path
        if os.path.exists(model_path):
            logging.info(
                "TDVP model already exists. Loading the old checkpoint and continuing training.")
            model = torch.load(model_path)
            if hasattr(model, "config"):
                self.learn_rate = model.config["lr"]
                logging.info(
                    f"Resetting the learning rate from {self.config['lr']} to {self.learn_rate}.")
                self.config["lr"] = self.learn_rate
            if hasattr(model, "best_val_acc_epoch") and hasattr(model, "epoch"):
                self.config["epoch"] = model.epoch
                self.epoch = model.epoch
                self.config['best_val_acc_epoch'] = model.best_val_acc_epoch
                logging.info(f"Epoch: {self.config['epoch']}")
                logging.info(
                    f"Best epoch: {self.config['best_val_acc_epoch']}")
            if hasattr(model, "best_val_acc"):
                self.config['best_val_acc'] = model.best_val_acc
                logging.info(f'Best acc: {self.config["best_val_acc"]}')
            else:
                logging.info(
                    f"Old checkpoint does not have a saved learning rate. Continuing with lr={learn_rate}")
            if hasattr(model, "wandb_name"):
                # Determine the previous wandb model name to continue logging to the same run
                self.wandb_name = model.wandb_name
            if hasattr(model, "wandb_id"):
                # Determine the previous wandb model name to continue logging to the same run
                self.wandb_id = model.wandb_id
        else:
            self.config.update({"ti_tdvp": self.config['ti']})
            model = CA_TDVP_V2(**self.config)

        # Using the GPUs
        if self.config['cuda']:
            logging.info("Using CUDA option.")
            model.to('cuda')

        self.num_model_params = sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)
        self.config["num_model_params"] = self.num_model_params

        model.config = self.config
        self.model = model

    def init_loss(self):
        if self.config['loss'] == "mae":
            if self.config["gaussian_loss_sigma"] > 0:
                def loss(labels, preds):
                    R = torch.randn(labels.shape, device=self.device) * \
                        self.config["gaussian_loss_sigma"] + 1.0
                    return torch.mean(torch.abs(labels-preds)*R)
            else:
                def loss(labels, preds):
                    return torch.mean(torch.abs(labels-preds))
            return loss
        elif self.config['loss'] == "mse":
            if self.config["gaussian_loss_sigma"] > 0:
                def loss(labels, preds):
                    R = torch.randn(labels.shape, device=self.device) * \
                        self.config["gaussian_loss_sigma"] + 1.

                    l = torch.pow(torch.abs(labels-preds), 2.0)*R
                    # Catching nans
                    nans = torch.isnan(l)
                    if nans.any():
                        logging.warning("Catching nans")
                        l[nans] = 0
                    return torch.mean(l)
                return loss
            else:
                return torch.nn.MSELoss()
        else:
            raise Exception("Loss not implemented!")

    def get_wandb_project(self, *args, **kwargs):
        return kwargs["project_name"]

    def get_model_name(self, *args, **kwargs):
        modelname = f"{self.prefix}CA_r{kwargs['rule']}_{kwargs['rule_j']}_D{kwargs['D']}n{kwargs['n']}crop{kwargs['crop']}Dtdvp{kwargs['D']}nrep{kwargs['nrep']}ntdvp{kwargs['ntdvp']}s{self.seed}{kwargs['activation']}e{kwargs['eps_tdvp']}r{kwargs['residual']}"
        if self.permute:
            modelname += "p"
        if self.ti:
            modelname += "ti"
        if self.random_crop:
            modelname += "r"
        return modelname

    def save_checkpoint(self):
        modelname = f"model_checkpoint.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(wandb.run.dir, modelname))

    def load_checkpoint(self):
        modelname = f"model_checkpoint.pt"
        checkpoint = torch.load(os.path.join(wandb.run.dir, modelname))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def check_loss(self, loss_list, train_loss, epoch, acc):
        handle_loss_decrease = self.config["handle_loss_decrease"]
        if handle_loss_decrease < 0.5:
            return False
        navg = 20
        if len(loss_list) < navg:
            return False
        ll = loss_list[-navg:]
        loss_mean = np.mean(ll)
        loss_std = np.std(ll)
        if (train_loss-loss_mean) < 2*loss_std:
            if (epoch - self.last_checkpoint_epoch) >= navg:
                self.save_checkpoint()
                self.last_checkpoint_epoch = epoch
                logging.info(f"Updating a checkpoint at epoch {epoch}.")
            return False
        logging.info(
            f"Reverting the model at epoch {epoch}. Metrics: loss {train_loss}, acc {acc}.")
        self.load_checkpoint()
        return True

    def pca_entropy(self, Aleft, Aright, idelta=2):
        pca = PCA()
        n = len(Aleft)
        imin = np.max([0, n//2-idelta])
        imax = np.min([n-1, n//2+idelta])
        Als = torch.cat(Aleft[imin:imax], dim=0)
        if self.cyclic:
            Ars = torch.cat(Aright[imin:imax], dim=0)
            A = torch.einsum("bti,bjt->bij", Als, Ars)
            A = torch.reshape(A, [A.shape[0], -1])
        else:
            A = Als
        pca.fit(A.clone().detach().numpy())
        # S = pca.singular_values_
        # S = S/np.sum(S)
        S = pca.explained_variance_ratio_
        return np.sum(-S*np.log(S))

        # return pca.n_components_

        # print("A.shape",A.shape)
        # U, S, V = torch.pca_lowrank(A, q=None, center=True, niter=2)
        # print("S.shape",S.shape)
        # print()
        # S = S/torch.linalg.norm(S)

    def train(self):
        run = self.train_wandb()

        loss_list = []

        # Let's start training!
        init_epoch = self.epoch
        logging.info("Start training!")
        max_acc = 0
        corr = []
        state = []
        ac_val = []
        ds_val = []
        ac_train = []
        ds_train = []

        a_list = []
        b_list = []
        scal_list = []
        gen_epoch = -1
        train_epoch = -1
        for epoch_num in range(init_epoch, self.num_epochs+1):
            train_loss, sample_acc, val_acc, domain_size, nrm, train_loss_main, train_loss_l1, pca = self.train_step(
                epoch_num)

            # Handling sudden decrease of the loss
            if self.check_loss(loss_list, train_loss, epoch_num, val_acc):
                train_loss, sample_acc, val_acc, domain_size, nrm, train_loss_main, train_loss_l1, pca = self.train_step(
                    epoch_num)

            loss_list.append(train_loss)

            ac, sacc, ds, pca_test = self.evaluate(save_model=False)

            ac_val.append(ac)
            ds_val.append(ds)
            ac_train.append(val_acc)
            ds_train.append(domain_size)

            if self.config["save_param_history"]:
                m = self.model.multi_tdvp.tdvps[0]

                aa = m.A.detach().clone().numpy()
                bb = m.B.detach().clone().numpy()
                scal = m.scale.detach().clone().numpy()
                a_list.append(aa)
                b_list.append(bb)
                scal_list.append(scal)

            # cr, st, ac100, ds100 = self.evaluate_corr(
            #     100, ntest=self.config["ntest"])

            cr = None
            st = None
            ac100, sacc100, ds100, pca_test_100 = self.evaluate(
                n=100, ntest=self.config["ntest"], save_model=False)

            if cr is not None:
                corr.append(cr)
                state.append(st)

            max_acc = np.max([max_acc, val_acc])

            if (self.killer.kill_now or self.stop_patience < epoch_num-self.best_val_acc_epoch):
                logging.warning(
                    f"Early stopping in epoch {epoch_num}. The accuracy did not increase since epoch {self.best_val_acc_epoch}.")
                break

            lr = self.scheduler._last_lr[0]
            wandb_logs = {'loss': train_loss, 'loss_main': train_loss_main, 'loss_l1': train_loss_l1, 'sample_accuracy': sample_acc,
                          'acc_train': val_acc, 'epoch': epoch_num, "lr": lr, "ds_train": domain_size, "norm": nrm, "acc_test": ac,
                          "ds_test": ds, "acc_test100": ac100, "ds_test100": ds100, "pca/train": pca, "pca/test": pca_test, "pca/test100": pca_test_100}

            logging.info(
                f"epoch {epoch_num:{8}} | loss {train_loss:{8}.{5}} | value acc {val_acc:{8}.{5}} | sample acc {sample_acc:{8}.{2}} | domain size {domain_size:{8}.{2}} | norm {nrm:{8}.{2}} | lr {lr:{8}.{7}}")

            if self.monitor_entropy:
                ent = entropy(self.model)
                wandb_logs["entropy"] = ent

                logging.info(f"Entropy: {ent}")

            if self.wandb:
                run.log(wandb_logs)

            if np.isnan(ac):
                logging.warning(
                    f"Stop training. Accuracy is: {ac} in epoch {epoch_num}.")

                n = self.input_dim

                X = torch.tensor(
                    np.array([[int(c) for c in bin(i)[2:].zfill(n)] for i in range(2**n)]))

                X = torch.tensor(X, dtype=torch.float32)
                Y = self.rule_fun(X)

                preds, _ = self.model(X)
                int_preds = torch.minimum(torch.maximum(
                    self.torch_zero, torch.round(preds)),  self.torch_one)

                m = self.model.multi_tdvp.tdvps[0]

            if (gen_epoch > 0 and epoch_num > gen_epoch + 400):
                logging.info(
                    f"Stop training. Maximum accuracy until epoch {epoch_num} is {max_acc}.")
                break

            if train_epoch < 0 and domain_size > 0.99:
                train_epoch = epoch_num

            if gen_epoch < 0 and ds > 0.99:
                gen_epoch = epoch_num
                logging.info(
                    f"Test accuracy {ac} setting gen_epoch to {gen_epoch}")

        running_time = int(time.time()-self.start_time)
        logging.warning(
            f"Ending training. Running time: {running_time/3600.:.3f} hours, Max training time: {self.max_training_hours} hours.")

        self.update_saved_model()
        # self.run_continue_script()

        print("\n")
        logging.info("EVALUATION ")
        acc_min = 1.0
        eval_acc = []
        for n in range(self.config['eval_nmin'], self.config['eval_nmax']+1):
            acc_val, acc_samp, domain_size, _ = self.evaluate(
                n, save_model=False)
            eval_acc.append([n, acc_val])
            acc_min = np.min([acc_val, acc_min])

        if self.wandb:
            run.log({"corr": np.array(corr), "state": np.array(
                state), "min_eval_acc": acc_min, "gen_epoch": gen_epoch, "train_epoch": train_epoch})
            table = wandb.Table(data=eval_acc, columns=["n", "acc_val"])
            run.log({"evaluation": wandb.plot.line(table,
                                                   "n", "acc_val", title="Evaluation accuracy")})
            run.finish()

        if self.config["save_param_history"] and self.wandb:
            print("Saving model history", run.dir)
            A = np.array(a_list)
            B = np.array(b_list)
            scal = np.array(scal_list)
            np.save(os.path.join(run.dir, "A.npy"), A)
            np.save(os.path.join(run.dir, "B.npy"), B)
            np.save(os.path.join(run.dir, "scale.npy"), scal)
        return acc_min, np.array(eval_acc)

    def train_step(self, epoch_num):
        self.model.train()
        running_loss = 0.
        running_loss_main = 0.
        running_loss_l1 = 0.
        running_val_acc = 0.
        running_samp_acc = 0.
        running_domain_size = 0
        running_nrm = 0
        running_pca = 0
        ibatch = 0
        for inputs, labels in self.train_ds:
            if self.killer.kill_now:
                break

            if self.random_crop:
                nmin = self.config["nmin"]
                dims = inputs.shape
                size = dims[1]
                ds = size-nmin
                size_cropped = np.random.randint(ds+1)+nmin
                dx = np.random.randint(size-size_cropped+1)
                inputs = inputs[:, dx:dx+size_cropped]
                labels = self.rule_fun(inputs)

            if self.killer.kill_now:
                break

            # TODO move the inds to the model...
            if self.inds is not None:
                inputs = inputs[:, self.inds, :]

            if self.killer.kill_now:
                break

            # Call our model to get logit scores and predictions
            if self.cuda:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

            preds, epreds, Als, Ars = self.model(inputs)

            nrm = torch.linalg.norm(epreds, dim=-1)

            int_preds = torch.minimum(torch.maximum(
                self.torch_zero, torch.round(preds)),  self.torch_one)

            if self.din == self.dout:
                elabels = self.embedding(labels)
                loss = self.loss_fun(elabels, epreds)
            else:
                loss = self.loss_fun(labels, preds)

            loss_main = loss.detach().numpy()

            if self.config['nrm_lambda'] > 0:
                loss += self.config['nrm_lambda'] * \
                    torch.abs(torch.mean((nrm-1)))

            # Compute L1 loss component
            loss_l1 = 0
            if self.l1 > 0:
                l1_weight = self.l1
                l1_parameters = []
                for parameter in self.model.parameters():
                    l1_parameters.append(parameter.view(-1))
                loss_l1 = l1_weight * torch.abs(torch.cat(l1_parameters)).sum()
                loss = loss + loss_l1

            # Adding entropy loss
            if self.ent_lambda != 0 and epoch_num > 0:
                if self.ent_mode == "center":
                    ent = entropy(self.model)
                    loss += self.ent_lambda*ent
                else:
                    ent = torch.mean(torch.stack(entropies(self.model)))
                    loss += self.ent_lambda*ent

            acc_val = torch.abs(int_preds-labels)
            acc_val = 1.0-torch.mean(acc_val)
            acc_samp = torch.sum(torch.abs(int_preds-labels), axis=1) < 0.1
            acc_samp = torch.mean(acc_samp.double())
            domain_size = average_domain_size(labels, int_preds)

            with torch.no_grad():
                running_loss += loss
                running_loss_main += loss_main
                running_loss_l1 += loss_l1
                running_samp_acc += acc_samp
                running_val_acc += acc_val
                running_domain_size += domain_size
                running_nrm += torch.mean(nrm)
                running_pca += self.pca_entropy(Als, Ars, idelta=3)
                ibatch += 1

            if self.killer.kill_now:
                break

            # Backpropagate and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        val_acc = running_val_acc/ibatch
        samp_acc = running_samp_acc/ibatch
        train_loss = running_loss/ibatch
        train_loss_main = running_loss_main/ibatch
        train_loss_l1 = running_loss_l1/ibatch
        domain_size = running_domain_size/ibatch
        nrm = running_nrm/ibatch
        pca = running_pca/ibatch

        if self.cuda:
            val_acc = val_acc.cpu()
            samp_acc = samp_acc.cpu()
            train_loss = train_loss.cpu()
            train_loss_main = train_loss_main.cpu()
            train_loss_l1 = train_loss_l1.cpu()
            domain_size = domain_size.cpu()
            nrm = nrm.cpu()

        # Learning rate scheduler
        # self.scheduler.step(val_acc)

        # Increasing the number of trained epochs
        self.epoch += 1

        # Checkpointing the model
        # This should be disabled in the tuner but can be enabled in the trainer
        self.save_best(val_acc, epoch_num)
        return train_loss, samp_acc, val_acc, domain_size, nrm, train_loss_main, train_loss_l1, pca

    def evaluate_corr(self, n=None, ntest=1):
        rule = self.rule
        rule_j = self.rule_j
        cyclic = hasattr(self, "cyclic") and self.cyclic

        if n == None:
            Xtest = self.Xtest
            Ytest = self.Ytest
        else:
            Xtest = torch.tensor(
                np.round(np.random.rand(ntest, n)), dtype=torch.float32)
            Ytest = self.rule_fun(Xtest)

            if self.cuda:
                Xtest = Xtest.to(self.device)
                Ytest = Ytest.to(self.device)

        preds, _, Als, Ars = self.model(Xtest)
        int_preds = torch.minimum(torch.maximum(
            self.torch_zero, torch.round(preds)),  self.torch_one)

        acc_val = torch.abs(int_preds-Ytest)
        acc_val = 1.0-torch.mean(acc_val)
        domain_size = average_domain_size(Ytest, int_preds)
        pca = self.pca_entropy(Als, Ars, idelta=5)

        acc_val = acc_val.cpu().detach().numpy()
        domain_size = domain_size.cpu().detach().numpy()

        corr, state = correlations(Ytest, int_preds)

        # corr = corr.cpu().detach().numpy()
        return corr, state, acc_val, domain_size

    def evaluate(self, n=None, ntest=1, save_model=False):
        rule = self.rule
        rule_j = self.rule_j

        cyclic = hasattr(self, "cyclic") and self.cyclic

        if n == None:
            Xtest = self.Xtest
            Ytest = self.Ytest
        else:
            Xtest = torch.tensor(
                np.round(np.random.rand(ntest, n)), dtype=torch.float32)
            Ytest = self.rule_fun(Xtest)

            if self.cuda:
                Xtest = Xtest.to(self.device)
                Ytest = Ytest.to(self.device)

        preds, _, Als, Ars = self.model(Xtest)
        int_preds = torch.minimum(torch.maximum(
            self.torch_zero, torch.round(preds)),  self.torch_one)

        acc_val = torch.abs(int_preds-Ytest)
        acc_val = 1.0-torch.mean(acc_val)
        acc_samp = torch.sum(torch.abs(int_preds-Ytest), axis=1) < 0.1
        acc_samp = torch.mean(acc_samp.double())
        domain_size = average_domain_size(Ytest, int_preds)
        pca = self.pca_entropy(Als, Ars, idelta=5)

        # logging.info(
        #     f"n {n:{4}} | value acc {acc_val:{8}.{5}} | sample acc {acc_samp:{8}.{2}} | domain size {domain_size:{8}.{2}}")

        if save_model:
            modelname = f"model"
            torch.save(self.model, os.path.join(wandb.run.dir, modelname))

        acc_val = acc_val.cpu().detach().numpy()
        acc_samp = acc_samp.cpu().detach().numpy()
        domain_size = domain_size.cpu().detach().numpy()
        return float(acc_val), float(acc_samp), float(domain_size), pca

    def arg_parser(self):
        parser = super(CellularAutomataTrainerCLI_V2, self).arg_parser()
        parser.add_argument('--Dtdvp',
                            default=4,
                            type=int,
                            help='Bond dimension of the MPO.')
        parser.add_argument('--dout',
                            default=1,
                            type=int,
                            help='Output feature size. 1 (default)')
        parser.add_argument('--cyclic',
                            default=1,
                            type=int,
                            help='If enabled we use closed boundary conditions. 0 (default), 1')
        parser.add_argument('--scale',
                            default=1.0,
                            type=int,
                            help='If enabled we use scale the left and the right boundary. Provided scale is the initial parameter and is train during training. If scale<=0 scaling is not applied. 1.0 (default)')
        parser.add_argument('--n',
                            default=8,
                            type=int,
                            help='Size of the samples.')
        parser.add_argument('--crop',
                            default=8,
                            type=int,
                            help='Mean size of the cropped samples. Only used if aut_random_crop is enabled.')
        parser.add_argument('--rule',
                            default=30,
                            type=int,
                            help='CA rule to use.')
        parser.add_argument('--rule_j',
                            default=1,
                            type=int,
                            help='Interaction range or effective interaction range.')
        parser.add_argument('--nrep',
                            default=1,
                            type=int,
                            help='Repetition of each MPO layer.')
        parser.add_argument('--ntdvp',
                            default=1,
                            type=int,
                            help='Number of MPO layers')
        parser.add_argument('--activation',
                            default="sigmoid",
                            type=str,
                            help='Activation of the MPO layers.')
        parser.add_argument('--handle_loss_decrease',
                            default=0,
                            type=int,
                            help='If enabled we check if the loss decreases suddenly and restore 10 epochs back.')
        parser.add_argument('--project_name',
                            default="grokking_v2",
                            type=str,
                            help='Activation of the MPO layers.')
        parser.add_argument('--dataset_mode',
                            default="all",
                            type=str,
                            help='What type of dataset is used. all (constructs all possible inputs), random (default - takes a random inputs of size ntrain x n)')
        parser.add_argument('--dtype',
                            default="float32",
                            type=str,
                            help='Precision of the model.')
        parser.add_argument('--lr_class',
                            default=0,
                            type=float,
                            help='Learning rate for the linear classifier matrix B. 0.001 (Default)')
        parser.add_argument('--loss',
                            default="mse",
                            type=str,
                            help='Loss function: mae (default), mse.')
        # parser.add_argument('--stddev',
        #                     default=1e-6,
        #                     type=float,
        #                     help='Standard deviation of the initial condition.')
        parser.add_argument('--l1',
                            default=0.001,
                            type=float,
                            help='L1 normalization of the TDVP kernels')
        parser.add_argument('--bufs',
                            default=512,
                            type=int,
                            help='Buffer size for reshufling')
        parser.add_argument('--nmin',
                            default=8,
                            type=int,
                            help='Minimum example size')
        parser.add_argument('--nmax',
                            default=8,
                            type=int,
                            help='Maximum example size')
        parser.add_argument('--eval_nmin',
                            default=256,
                            type=int,
                            help='Minimum example size')
        parser.add_argument('--eval_nmax',
                            default=256,
                            type=int,
                            help='Maximum example size')
        parser.add_argument('--residual',
                            default=0,
                            type=int,
                            help='Enables the residual layers.')
        parser.add_argument('--mode',
                            default="sequential",
                            type=str,
                            help='Evaluation mode of the TDVP layer.')
        parser.add_argument('--trainable_A',
                            default=0,
                            type=int,
                            help='Determines if the boundary matrices are trainable. If they are not they are chosen to be random matrices with variance 1. 1 (default), 0')
        parser.add_argument('--out_norm',
                            default="none",
                            type=str,
                            help='Normalization of the TDVP output: none(default), L1, L2')
        parser.add_argument('--ti_tdvp',
                            action='store_true',
                            help='Translationary invariant TDVP layer.')
        parser.add_argument('--disable_tqdm',
                            action='store_true',
                            help='Disables tqdm monitoring of the training.')
        parser.add_argument('--eps_tdvp',
                            default=0.1,
                            type=float,
                            help='Epsilon for the initialization of the TDVP mpos. 0.000001 (default). Note that the actual values used in the calculation are calculated with ArcTan. So they are bounded for any eps. eps=100 produces a uniform distribution.')
        parser.add_argument('--gaussian_loss_sigma',
                            default=0,
                            type=float,
                            help='Multiply the example loss function with a gaussian random number with standard deviation gaussian_loss_sigma and mean 1. Only applied if gaussian_loss_sigma>0. 0 (default).')
        parser.add_argument('--nrm_lambda',
                            default=0,
                            type=float,
                            help='Strangth of the output norm regularizer. 1e-6 (default)')
        parser.add_argument('--save_param_history',
                            default=0,
                            type=int,
                            help='If enabled the history of model parameters is saved. 0 (default)')

        return parser

import torch
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torchvision.utils import make_grid, save_image
from engression.loss_func import energy_loss_two_sample
from engression.data.loader import make_dataloader

from .model import DPAmodel
from . import utils


class DPA(object):
    """Distributional Principal Autoencoder.

    Args:
        data_dim (int): data dimension.
        latent_dims (list or int): list of latent dimensions.
        num_layer (int, optional): number of layers. Defaults to 2.
        num_layer_enc (int, optional): number of layers for the encoder. Defaults to num_layer.
        hidden_dim (int, optional): hidden dimension. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        out_dim (int, optional): output dimension. Defaults to data_dim.
        condition_dim (int, optional): dimension of the conditioning set. Defaults to None.
        linear (bool, optional): whether to use linear encoder. Defaults to False.
        lin_dec (bool, optional): whether to use linear decoder when using a linear encoder. Defaults to True.
        lin_bias (bool, optional): whether to include a bias term when using linear models. Defaults to False.
        dist_enc (str, optional): distribution of the encoder. Defaults to "deterministic".
        dist_dec (str, optional): distribution of the decoder. Defaults to "stochastic".
        resblock (bool, optional): whether to use residual blocks or skip connections. Defaults to True.
        out_act (str, optional): output activation function to ensure bounded output. Defaults to None.
        bn_enc (bool, optional): whether to use batch norm in encoder. Defaults to False.
        bn_dec (bool, optional): whether to use batch norm in decoder. Defaults to False.
        encoder_k (bool, optional): whether to include k as an encoder input. Defaults to False.
        coef_match_latent (int, optional): coefficient of the energy loss on latents. Defaults to 0.
        lr (float, optional): learning rate. Defaults to 1e-4.
        num_epochs (int, optional): number of training epochs. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None.
        standardize (bool, optional): whether to standardize data. Defaults to False.
        beta (int, optional): beta in the energy loss. Defaults to 1.
        device (str, optional): device.
        dim1 (int, optional): first dimension of the image. Defaults to 192.
        dim2 (int, optional): second dimension of the image. Defaults to 288.
        seed (int, optional): random seed for reproduction. Defaults to 222.
    """
    def __init__(self, 
                 data_dim, latent_dims, num_layer=2, num_layer_enc=None, hidden_dim=100, noise_dim=100, 
                 out_dim=None, condition_dim=None, linear=False, lin_dec=True, lin_bias=False,
                 dist_enc="deterministic", dist_dec="stochastic", resblock=True, out_act=None, 
                 bn_enc=False, bn_dec=False, encoder_k=False, coef_match_latent=0,
                 lr=1e-4, num_epochs=500, batch_size=None, standardize=False, beta=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 dim1=192, dim2=288, seed=222):
        super().__init__()
        self.data_dim = data_dim
        if not isinstance(latent_dims, list):
            latent_dims = list(latent_dims)
        self.latent_dims = latent_dims
        self.latent_dim = latent_dims[0]
        self.num_levels = len(latent_dims)
        self.num_layer = num_layer
        self.num_layer_enc = num_layer_enc
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.out_dim = out_dim
        self.condition_dim = condition_dim
        self.linear = linear
        self.lin_dec = lin_dec
        self.lin_bias = lin_bias
        self.dist_enc = dist_enc
        self.dist_dec = dist_dec
        self.bn_enc = bn_enc
        self.bn_dec = bn_dec
        self.out_act = out_act
        self.dim1 = dim1
        self.dim2 = dim2
        
        self.encoder_k = encoder_k
        self.coef_match_latent = coef_match_latent
        self.match_latent = coef_match_latent > 0
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta
        
        if isinstance(device, str):
            if device == "gpu" or device == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device(device)
        self.device = device
        utils.check_for_gpu(self.device)
        
        self.standardize = standardize
        self.x_mean = None
        self.x_std = None
        
        self.loss_all_k = np.zeros(self.num_levels)
        self.loss_pred_all_k = np.zeros(self.num_levels)
        self.loss_var_all_k = np.zeros(self.num_levels)
        self.loss_all_k_test = np.zeros(self.num_levels)
        self.loss_pred_all_k_test = np.zeros(self.num_levels)
        self.loss_var_all_k_test = np.zeros(self.num_levels)
        self.energy_loss = None
        self.recon_mse = None
        
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.model = DPAmodel(data_dim=data_dim, latent_dim=self.latent_dim, out_dim=self.out_dim, condition_dim=condition_dim,
                              num_layer=num_layer, num_layer_enc=num_layer_enc, hidden_dim=hidden_dim, noise_dim=noise_dim, 
                              dist_enc=dist_enc, dist_dec=dist_dec, resblock=resblock, encoder_k=encoder_k,
                              bn_enc=bn_enc, bn_dec=bn_dec, out_act=out_act, 
                              linear=linear, lin_dec=lin_dec, lin_bias=lin_bias).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()

    def _standardize_data_and_record_stats(self, x, univar=False):
        self.x_mean = torch.mean(x, dim=0)
        if univar:
            self.x_std = x.std()
        else:
            self.x_std = torch.std(x, dim=0)
            self.x_std[self.x_std == 0] += 1e-5
        x_standardized = (x - self.x_mean) / self.x_std
        self.x_mean = self.x_mean.to(self.device)
        self.x_std = self.x_std.to(self.device)
        return x_standardized
        
    def standardize_data(self, x):
        if self.standardize:
            return (x - self.x_mean) / self.x_std
        else:
            return x
        
    def unstandardize_data(self, x):
        if self.standardize:
            return x * self.x_std + self.x_mean
        else:
            return x
                
    def train(self, x, x_te=None, c=None, c_te=None, num_epochs=None, num_pro_epoch=0, batch_size=None, 
              print_every_nepoch=100, print_all_k=True, 
              standardize=None, univar=False, lr=None,
              save_model_every=None, save_recon_every=0, n_recon=5, recon_color=True, save_dir="", save_loss=False,
              resume_epoch=None):
        """Fit the model.

        Args:
            x (torch.Tensor or DataLoader): training data.
            x_te (torch.Tensor, optional): test data. Defaults to None.
            c (torch.Tensor or DataLoader, optional): training conditioning variable. Defaults to None.
            c_te (torch.Tensor, optional): test conditioning variable. Defaults to None.
            num_epochs (int, optional): number of epochs. Defaults to None.
            num_pro_epoch (int, optional): number of progressive epochs. Defaults to 0.
            batch_size (int, optional): batch size. Defaults to None.
            print_every_nepoch (int, optional): print losses per print_every_nepoch epochs. Defaults to 100.
            print_all_k (bool, optional): print losses for all k. Defaults to True.
            standardize (bool, optional): whether to standardize data. Defaults to None.
            univar (bool, optional): univariate scaling in standardization. Defaults to False.
            lr (float, optional): learning rate. Defaults to None.
            save_model_every (int, optional): save model per save_model_every epochs. Defaults to None.
            save_recon_every (int, optional): save reconstructions per save_recon_every epochs. Defaults to 0.
            n_recon (int, optional): number of reconstructions to save. Defaults to 5.
            recon_color (bool, optional): colored reconstructions. Defaults to True.
            save_dir (str, optional): directory for saving the results. Defaults to "".
            save_loss (bool, optional): whether to save loss or not. Defaults to False.
            resume_epoch (int, optional): resume epoch. Defaults to None.
        """
        ## resume from checkpoints
        if resume_epoch is not None:
            print(f"Resume training from epoch {resume_epoch}")
            ckpt_dir = save_dir + "model_" + str(resume_epoch) + ".pt"
            self.model.load_state_dict(torch.load(ckpt_dir))
            start_epoch = resume_epoch
        else:
            start_epoch = 0
        
        if lr is not None:
            if lr != self.lr:
                self.lr = lr
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
        ## basic settings
        self.train_mode()
        if num_epochs is None:
            num_epochs = self.num_epochs
        if batch_size is None:
            batch_size = self.batch_size
        if standardize is not None:
            self.standardize = standardize
        
        if num_pro_epoch == 0: 
            num_pro_epoch = 1e-5   
        
        if torch.is_tensor(x):
            ## When inputs are torch tensors
            if self.standardize:
                x = self._standardize_data_and_record_stats(x, univar)
            
            if self.condition_dim is not None:
                # dummy encode y (todo for dataloader)
                c = utils.num2onehot(c)
                c_te = utils.num2onehot(c_te).to(self.device)
                
            if save_recon_every > 0:
                np.random.seed(222)
                eval_idx = np.random.permutation(x.size(0))
        
        else:
            ## When input is a dataloader
            if save_recon_every > 0:
                x_eval, c_eval = next(iter(x))
                
        if x_te is not None:
            # x_te = x_te[:batch_size]
            x_te = x_te.to(self.device)
            if self.standardize:
                x_te = self.standardize_data(x_te)
        if save_recon_every > 0:    
            x_eval = x[eval_idx[:n_recon]].to(self.device)
            c_eval = None if self.condition_dim is None else c[eval_idx[:n_recon]].to(self.device)
            x_te_eval = x_te[:n_recon]
            c_te_eval = None if self.condition_dim is None else c_te[:n_recon]
                
        ## for exporting results
        if save_model_every is not None or save_recon_every > 0 or save_loss:
            utils.make_folder(save_dir)
        if save_loss:
            log_file_name = os.path.join(save_dir, "log.txt")
            if resume_epoch is not None:
                log_file = open(log_file_name, "at")
            else:
                log_file = open(log_file_name, "wt")
        
        ## Start training
        if batch_size >= x.size(0)//2:
            raise ValueError("Not implemented.")
            # assert torch.is_tensor(x)
            # todo
        else:
            train_loader = make_dataloader(x, c, batch_size=batch_size, shuffle=True)
            print(f"Start training with {len(train_loader)} batches each of size {batch_size}.\n")
            if self.condition_dim is not None:
                c = utils.onehot2num(c)
            for epoch_idx in range(start_epoch, num_epochs):
                self.loss_all_k = np.zeros(self.num_levels)
                self.loss_pred_all_k = np.zeros(self.num_levels)
                self.loss_var_all_k = np.zeros(self.num_levels)
                for batch_idx, x_batch in enumerate(train_loader):
                    if c is None:
                        x_batch = x_batch[0]
                        c_batch = None
                    else:
                        x_batch, c_batch = x_batch
                        c_batch = c_batch.to(self.device)
                    x_batch = x_batch.to(self.device)
                    
                    k_max = min(int((epoch_idx + 1) // num_pro_epoch), self.num_levels - 1)
                    self.train_one_iter(x_batch, c_batch, k_max)
                    
                    if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0):
                        if batch_idx + 1 == (len(train_loader) - 1):
                            print_loss_str = self.print_loss(x_te, c_te, batch_idx, epoch_idx, k_max, print_all_k)
                            print(print_loss_str)
                            if save_loss:
                                log_file.write(print_loss_str + "\n")
                                log_file.flush()
                            self.train_mode()
                            
                if save_model_every is not None:
                    if (epoch_idx + 1) % save_model_every == 0:
                        torch.save(self.model.state_dict(), save_dir + "model_" + str(epoch_idx + 1) + ".pt")
                
                if save_recon_every > 0:
                    if (epoch_idx==0) or ((epoch_idx + 1) % save_recon_every == 0):
                        self.save_recon(x_eval, c_eval, k_max=k_max, n_row=n_recon, save_dir=save_dir + f"recon_tr{epoch_idx + 1}_k{k_max}.png", gen_sample_size=1, color=recon_color)
                        self.save_recon(x_te_eval, c_te_eval, k_max=k_max, n_row=n_recon, save_dir=save_dir + f"recon_te{epoch_idx + 1}_k{k_max}.png", gen_sample_size=1, color=recon_color)
                        self.train_mode()
        
        # self.model.eval()
        # with torch.no_grad():
        #     for k in range(self.num_levels):
        #         gen1 = self.model(x, k)
        #         gen2 = self.model(x, k)
        #         loss, s1, s2 = energy_loss_two_sample(x, gen1, gen2, verbose=True)
        #         self.loss_all_k[k] = loss.item()
        #         self.loss_pred_all_k[k] = s1.item()
        #         self.loss_var_all_k[k] = s2.item()
        
    def train_one_iter(self, x_batch, c_batch, k_max):
                
        self.model.zero_grad()
        losses = []
        for k in range(k_max + 1):
            return_latent = self.match_latent & (k == k_max)
            gen1 = self.model(x=x_batch, k=self.latent_dims[k], c=c_batch, return_latent=return_latent, double=True)
            if return_latent:
                gen1, gen2, z = gen1
            else:
                gen1, gen2 = gen1
            loss, s1, s2 = energy_loss_two_sample(x_batch, gen1, gen2, beta=self.beta, verbose=True)
            self.loss_all_k[k] += loss.item()
            self.loss_pred_all_k[k] += s1.item()
            self.loss_var_all_k[k] += s2.item()
            losses.append(loss)
        loss = sum(losses)
        if self.match_latent:
            z_gauss = torch.randn((gen1.size(0), self.latent_dim), device=self.device)
            indices = np.random.permutation(gen1.size(0))
            idx1 = indices[:(gen1.size(0) // 2)]
            idx2 = indices[(gen1.size(0) // 2):]
            z1 = z[idx1, :]
            z2 = z[idx2, :]
            z_gauss1, z_gauss2 = z_gauss.chunk(2, dim=0)
            if z1.size(0) != z2.size(0):
                z2 = z2[:-1]
                z_gauss1 = z_gauss1[:-1]
            loss_z, s1_z, s2_z = energy_loss_two_sample(z_gauss1, z1, z2, z_gauss2, verbose=True)
            # loss_z = z.mean(dim=1).pow(2).mean() + z.var(dim=1, correction=0).mean()
            loss = loss + loss_z * self.coef_match_latent
                
        loss.backward()
        self.optimizer.step()

    def print_loss(self, x_te, c_te, batch_idx, epoch_idx, k_max, print_all_k, printout=False):
        self.loss_all_k = self.loss_all_k / (batch_idx + 1)
        self.loss_pred_all_k = self.loss_pred_all_k / (batch_idx + 1)
        self.loss_var_all_k = self.loss_var_all_k / (batch_idx + 1)
        
        print_loss_str = f"[Epoch {epoch_idx + 1}] "
        if print_all_k:
            print_loss_str += ", ".join("{:.4f}".format(f) for f in self.loss_all_k[:(k_max + 1)]) + "\n"
            print_loss_str += " pred \t" + ", ".join("{:.4f}".format(f) for f in self.loss_pred_all_k[:(k_max + 1)]) + "\n"
            print_loss_str += " var \t" + ", ".join("{:.4f}".format(f) for f in self.loss_var_all_k[:(k_max + 1)]) + "\n"
        else:
            loss_mean = np.mean(np.array(self.loss_all_k)[:(k_max + 1)])
            loss_min = np.min(np.array(self.loss_all_k)[:(k_max + 1)])
            loss_s1_mean = np.mean(np.array(self.loss_pred_all_k)[:(k_max + 1)])
            loss_s2_mean = np.mean(np.array(self.loss_var_all_k)[:(k_max + 1)])
            print_loss_str += f" average loss {loss_mean:.4f}, {loss_s1_mean:.4f}, {loss_s2_mean:.4f}, min {loss_min:.4f}"
        
        if x_te is not None:
            self.eval_mode()
            with torch.autocast(device_type="cuda"):
                with torch.no_grad():
                    for k in range(k_max + 1):
                        gen1, gen2 = self.model(x=x_te, k=self.latent_dims[k], c=c_te, double=True)
                        loss, s1, s2 = energy_loss_two_sample(x_te, gen1, gen2, verbose=True)
                        self.loss_all_k_test[k] = loss.item()
                        self.loss_pred_all_k_test[k] = s1.item()
                        self.loss_var_all_k_test[k] = s2.item()
            self.train_mode()
            if print_all_k:
                print_loss_str += "(test)\t"
                print_loss_str += ", ".join("{:.4f}".format(f) for f in self.loss_all_k_test[:(k_max + 1)]) + "\n"
                print_loss_str += " pred \t" + ", ".join("{:.4f}".format(f) for f in self.loss_pred_all_k_test[:(k_max + 1)]) + "\n"
                print_loss_str += " var \t" + ", ".join("{:.4f}".format(f) for f in self.loss_var_all_k_test[:(k_max + 1)]) + "\n"
            else:
                loss_mean = np.mean(np.array(self.loss_all_k_test)[:(k_max + 1)])
                loss_min = np.min(np.array(self.loss_all_k_test)[:(k_max + 1)])
                loss_s1_mean = np.mean(np.array(self.loss_pred_all_k_test)[:(k_max + 1)])
                loss_s2_mean = np.mean(np.array(self.loss_var_all_k_test)[:(k_max + 1)])
                print_loss_str += f"; test average loss {loss_mean:.4f}, {loss_s1_mean:.4f}, {loss_s2_mean:.4f}, min {loss_min:.4f}"
        if printout:
            print(print_loss_str)
            return None
        return print_loss_str
            
    def plot_energy_loss(self, x=None, for_k=None, save_dir=None, xscale="symlog"):
        if for_k is None:
            for_k = self.latent_dims
        self.energy_loss = []
        self.loss_pred_all_k_test = []
        self.loss_var_all_k_test = []
        if x is None:
            plt.plot(for_k, self.loss_all_k[for_k], label="energy-loss")
            plt.plot(for_k, self.loss_pred_all_k[for_k], label="pred")
            plt.plot(for_k, self.loss_var_all_k[for_k], label="var")
        else:
            self.model.eval()
            if self.standardize:
                x = self.standardize_data(x)
            with torch.no_grad():
                for k in for_k:
                    gen1 = self.model(x, k=k)
                    gen2 = self.model(x, k=k)

                    if self.standardize:
                        gen1 = self.unstandardize_data(gen1)
                        gen2 = self.unstandardize_data(gen2)
                        x = self.unstandardize_data(x)
                    
                    # evaluate the energy score on the original data scale
                    loss, s1, s2 = energy_loss_two_sample(x, gen1, gen2, verbose=True)
                    self.energy_loss.append(loss.item())
                    self.loss_pred_all_k_test.append(s1.item())
                    self.loss_var_all_k_test.append(s2.item())
            plt.plot(for_k, self.energy_loss, label="loss")
            plt.plot(for_k, self.loss_pred_all_k_test, label="s1")
            plt.plot(for_k, self.loss_var_all_k_test, label="s2")
            plt.xscale(xscale); plt.xticks(for_k, for_k)

        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches="tight")
        plt.legend()
        plt.show()
        
    def plot_mse(self, x, c=None, for_k=None, gen_sample_size=100, verbose=False, save_dir=None, xscale="symlog"):
        # MSE loss
        self.recon_mse = []
        if for_k is None:
            for_k = self.latent_dims
        if not isinstance(for_k, list):
            for_k = range(for_k + 1)
        for k in for_k:
            recon_loss = self.reconstruct(x, c, k, gen_sample_size=gen_sample_size, mean=True, verbose=False, return_loss=True)
            self.recon_mse.append(recon_loss)
            if verbose:
                print(k, recon_loss)
        plt.plot(for_k, np.array(self.recon_mse))
        plt.xscale(xscale); plt.xticks(for_k, for_k)
        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches="tight")
        plt.show()
    
    def plot_2d_embedding(self, x, c, save_dir=None):
        z = self.encode(x).detach().cpu()
        model_id = c[:,0]
        lut = np.sort(np.unique(model_id)) 
        model_id = np.searchsorted(lut, model_id)
        data = pd.DataFrame({"z1": z[:,0], "z2": z[:,1], "month": c[:,1], "model": model_id})
        plt.rcParams.update({"font.size": 14})
        plt.rcParams["figure.figsize"] = (6,6)
        g = sns.scatterplot(data, x="z1", y="z2", hue="month", style="model", palette="rainbow", legend="full") 
        handles, labels = g.get_legend_handles_labels()
        for i,label in enumerate(labels):
            if label == "month":
                continue
            if label == "model":
                break
            handles[i] = mpatches.Patch(color=handles[i].get_mfc())
        plt.legend(handles, labels, bbox_to_anchor=(1.01, 1)); plt.xlabel(r"$Z_1$"); plt.ylabel(r"$Z_2$"); 
        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches="tight")
        plt.show(); plt.close()
        
    @torch.no_grad()
    def reconstruct(self, x, c=None, k=None, mean=False, gen_sample_size=100, verbose=True, return_loss=False, in_training=False):
        self.model.eval()
        # if k is None:
        #     k = self.latent_dim
        if not in_training:
            x_raw = x.clone()
            if self.standardize:
                x = self.standardize_data(x)
        
        x_recon = self.model.reconstruct(x, c, k, mean, gen_sample_size, verbose=False)
        if self.standardize:
            x = self.unstandardize_data(x)
            x_recon = self.unstandardize_data(x_recon)
        if in_training:
            x_recon = x_recon.view(x_recon.size(0), 1, self.dim1, self.dim2)            
        if return_loss and not in_training:
            recon_loss = (x_recon - x_raw).pow(2).mean().item()
            if verbose:
                print(k, recon_loss)
            return recon_loss
        else:
            return x_recon
                    
    def save_recon(self, x, c=None, k_max=None, gen_sample_size=100, n_row=5, save_dir="", color=True):
        # only used during training
        recon = []
        k_max = self.num_levels if k_max is None else k_max + 1
        for k in range(k_max):
            recon.append(self.reconstruct(x, c, self.latent_dims[k], mean=False, gen_sample_size=gen_sample_size, 
                                          verbose=False, in_training=True).cpu())
        recon = torch.cat(recon, dim=0)
        if self.standardize:
            x = self.unstandardize_data(x)
        x = x.cpu()
        if len(x.shape) <= 2:
            x = x.view(x.size(0), 1, self.dim1, self.dim2)
        if len(recon.shape) <= 2:
            recon = recon.view(recon.size(0), 1, self.dim1, self.dim2)
        recon = torch.cat([x, recon], dim=0)
        if color:
            recon = torch.clamp(recon, torch.quantile(x, 0.005).item(), torch.quantile(x, 0.995).item())
            plt.matshow(make_grid(recon, nrow=n_row).permute(1, 2, 0)[:,:,0], cmap="rainbow"); plt.axis("off"); 
            plt.savefig(save_dir, bbox_inches="tight", pad_inches=0, dpi=300); plt.close()
        else:
            save_image(recon, save_dir, nrow=n_row, normalize=False, scale_each=False)
    
    @torch.no_grad()
    def encode(self, x, k=None, mean=True, gen_sample_size=100, in_training=True):
        self.eval_mode()
        if not in_training:
            if self.standardize:
                x = self.standardize_data(x)
        z = self.model.encode(x, k, mean, gen_sample_size)
        self.train_mode()
        return z

    @torch.no_grad()
    def decode(self, z, c=None, mean=True, gen_sample_size=100):
        self.eval_mode()
        samples = self.model.decode(z, c, mean, gen_sample_size)
        if self.standardize:
            samples = self.unstandardize_data(samples)
        self.train_mode()
        return samples 


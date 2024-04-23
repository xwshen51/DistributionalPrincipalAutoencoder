import torch
import torch.nn as nn
import sys
import numpy as np
from engression.models import StoNet, StoLayer
from engression.data.loader import make_dataloader


class DPAmodel(nn.Module):
    def __init__(self, data_dim=2, latent_dim=10, out_dim=None, condition_dim=None,
                 num_layer=3, num_layer_enc=None, hidden_dim=500, noise_dim=None, 
                 dist_enc="deterministic", dist_dec="deterministic", resblock=True,
                 encoder_k=False, bn_enc=False, bn_dec=False, out_act=None, 
                 linear=False, lin_dec=True, lin_bias=True):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        if out_dim is None:
            out_dim = data_dim
        self.out_dim = out_dim
        self.condition_dim = condition_dim
        self.num_layer = num_layer
        if num_layer_enc is None:
            num_layer_enc = num_layer
        self.num_layer_enc = num_layer_enc
        self.hidden_dim = hidden_dim
        noise_dim_enc = 0 if dist_enc == "deterministic" else noise_dim
        noise_dim_dec = 0 if dist_dec == "deterministic" else noise_dim
        self.noise_dim = noise_dim
        self.noise_dim_enc = noise_dim_enc
        self.noise_dim_dec = noise_dim_dec
        self.dist_enc = dist_enc
        self.dist_dec = dist_dec
        self.out_act = out_act
        self.linear = linear
        self.lin_dec = lin_dec
        self.encoder_k = encoder_k
        
        if not linear:
            self.encoder = StoNet(data_dim, latent_dim, num_layer_enc, hidden_dim, noise_dim_enc, bn_enc, resblock=resblock)
            if condition_dim is not None: # conditional decoder
                latent_dim = latent_dim + condition_dim
            self.decoder = StoNet(latent_dim, out_dim, num_layer, hidden_dim, noise_dim_dec, bn_dec, out_act, resblock)     
        else:
            self.encoder = nn.Linear(data_dim, latent_dim, bias=lin_bias)
            if lin_dec:
                if noise_dim_dec == 0:
                    self.decoder = nn.Linear(latent_dim, out_dim, bias=lin_bias)
                    if out_act == "relu":
                        self.decoder = nn.Sequential(*[self.decoder, nn.ReLU(inplace=True)])
                    elif out_act == "sigmoid":
                        self.decoder = nn.Sequential(*[self.decoder, nn.Sigmoid()])
                else:
                    self.decoder = StoLayer(latent_dim, out_dim, noise_dim_dec, out_act=out_act)
            else:
                self.decoder = StoNet(latent_dim, out_dim, num_layer, hidden_dim, noise_dim_dec, bn_dec, out_act, resblock)
        
        if self.encoder_k:
            self.k_embed_layer = nn.Linear(self.latent_dim, self.data_dim*2)
    
    def get_k_embedding(self, k, x=None):
        k_emb = torch.ones(1, self.latent_dim)
        k_emb[:, k:].zero_()
        if x is not None:
            k_emb = k_emb.to(x.device)
            gamma, beta = self.k_embed_layer(k_emb).chunk(2, dim=1)
            k_emb = gamma * x + beta
        return k_emb
    
    def encode(self, x, k=None, mean=True, gen_sample_size=100, in_training=False):
        if k is None:
            k = self.latent_dim
        if self.encoder_k:
            x = self.get_k_embedding(k, x)
        if in_training:
            return self.encoder(x)
        if self.dist_enc == "deterministic":
            gen_sample_size = 1
        if mean:
            z = self.encoder.predict(x, sample_size=gen_sample_size)
        else:
            z = self.encoder.sample(x, sample_size=gen_sample_size)
            if gen_sample_size == 1:
                z = z.squeeze(len(z.shape) - 1)
        return z[:, :k]
        
    def decode(self, z, c=None, mean=True, gen_sample_size=100):
        if z.size(1) != self.latent_dim:
            z_ = torch.randn((z.size(0), self.latent_dim - z.size(1)), device=z.device)
            z = torch.cat([z, z_], dim=1)
        if c is not None:
            z = torch.cat([z, c], dim=1)
        if self.dist_enc == "deterministic":
            gen_sample_size = 1
        if mean:
            x = self.decoder.predict(z, sample_size=gen_sample_size)
        else:
            x = self.decoder.sample(z, sample_size=gen_sample_size)
        return x
    
    @torch.no_grad()
    def reconstruct_onebatch(self, x, c=None, k=None, mean=False, gen_sample_size=100):
        if gen_sample_size > 1 and self.dist_enc == "deterministic" and self.dist_dec == "deterministic":
            print("The model is deterministic. Consider setting `gen_sample_size` to 1.")
        if k is None:
            k = self.latent_dim
        if gen_sample_size == 1:#self.dist_enc == "deterministic" and self.dist_dec == "deterministic" or 
            return self.forward(x, c, k).detach()
        x_rep = x.repeat(gen_sample_size, *[1]*(len(x.shape) - 1))
        samples = self.forward(x_rep, c, k).detach()
        del x_rep
        expand_dim = len(samples.shape)
        samples = samples.unsqueeze(expand_dim)
        samples = list(torch.split(samples, x.size(0)))
        samples = torch.cat(samples, dim=expand_dim)
        if mean:
            mean_recon = samples.mean(dim=len(samples.shape) - 1)
            return mean_recon
        else:
            return samples
    
    def reconstruct_batch(self, x, c=None, k=None, mean=False, gen_sample_size=100, batch_size=None):
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, c, batch_size=batch_size, shuffle=False)
            results = []
            for x_batch in test_loader:
                if c is None:
                    x_batch = x_batch[0]
                    c_batch = None
                else:
                    x_batch, c_batch = x_batch
                results.append(self.reconstruct_onebatch(x_batch, c_batch, k, mean, gen_sample_size))
            results = torch.cat(results, dim=0)
        else:
            results = self.reconstruct_onebatch(x, c, k, mean, gen_sample_size)
        return results
        
    def reconstruct(self, x, c=None, k=None, mean=False, gen_sample_size=100, verbose=True):
        batch_size = x.shape[0]
        while True:
            try:
                results = self.reconstruct_batch(x, c, k, mean, gen_sample_size, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if verbose:
                        print("Out of memory; reduce the batch size to {}".format(batch_size))
        return results
        
    def forward(self, x, c=None, k=None, gen_sample_size=None, return_latent=False, device=None, double=False):
        if k is None:
            k = self.latent_dim
        if self.encoder_k:
            x = self.get_k_embedding(k, x)
        if double:
            z = self.encode(x, in_training=True)
            z1 = z.clone()
            if return_latent:
                z_ = z.clone()
            z[:, k:].normal_(0, 1)
            x1 = self.decoder(z)
            z1[:, k:].normal_(0, 1)
            x2 = self.decoder(z1)
            if return_latent:
                return x1, x2, z_
            else:
                return x1, x2
        else:
            if x is not None and k > 0:
                z = self.encode(x, in_training=True)
                if return_latent:
                    z_ = z.clone()
                z[:, k:].normal_(0, 1)
            else:
                if return_latent:
                    z_ = self.encode(x, in_training=True)
                if gen_sample_size is None:
                    gen_sample_size = x.size(0)
                if device is None:
                    device = x.device if x is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                z = torch.randn((gen_sample_size, self.latent_dim), device=device)
            if self.condition_dim is not None and c is not None:
                z = torch.cat([z, c], dim=1)
            x = self.decoder(z)
            if return_latent:
                return x, z_
            else:
                return x
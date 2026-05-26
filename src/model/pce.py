from torch import nn
import torch
import numpy as np
from torch.nn import init
from .base import BaseModule
from ..modules.patch_nce import PatchNCELoss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        if self.power not in [1, 2]:
            self.forward = self.identity

    def forward(self, x):
        norm = x.norm(p=self.power, dim=1, keepdim=True)
        return x.div(norm + 1e-7)

    def identity(self, x):
        return x


class PatchLatentProjector(nn.Module):
    def __init__(
        self,
        init_type="normal",
        init_gain=0.02,
        nc=256,
        nce_norm=2,
        input_ncs=[16, 64, 256],
        num_feats=3,
    ):
        super().__init__()
        self.l2norm = Normalize(nce_norm)
        self.nc = nc
        self.init_type = init_type
        self.init_gain = init_gain
        self.mlps = nn.ModuleList()
        self.nce_norm = nce_norm

        for input_nc, _ in zip(input_ncs, range(num_feats)):
            mlp = nn.Sequential(
                *[
                    nn.Linear(input_nc, self.nc),
                    nn.ReLU(),
                    nn.Linear(self.nc, self.nc),
                ]
            )
            self.mlps.append(mlp)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if hasattr(m, "weight") and m.weight is not None:
            if self.init_type == "normal":
                init.normal_(m.weight.data, 0.0, self.init_gain)
            elif self.init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif self.init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=self.init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % self.init_type)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]

            device = feat.device
            # [bs, H*W, num_channels]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=device)
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]

                if patch_id.ndim == 2 and patch_id.size(0) == B:
                    x_sample = torch.stack([f[p] for f, p in zip(feat_reshape, patch_id)]).flatten(0, 1)
                else:
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []

            mlp = self.mlps[feat_id]
            x_sample = mlp(x_sample)
            return_ids.append(patch_id)

            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample_shape = [B, x_sample.shape[-1], H, W]
                x_sample = x_sample.permute(0, 2, 1).reshape(x_sample_shape)

            x_sample = x_sample.view(B, -1, x_sample.size(-1))

            return_feats.append(x_sample)

        return return_feats, return_ids


class LatentEncoder(nn.Module):
    def __init__(self, latent_size=8):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size_stride = [
            (self.latent_size // 4, self.latent_size // 8),
            (self.latent_size // 2, self.latent_size // 4),
            (self.latent_size, self.latent_size // 2),
        ]  # patch size: 2,4,8, stride: 1,2,4
        self.output_channels = [
            16,
            64,
            256,
        ]  # later will all be 256 afeter go through projector

    def forward(self, x):
        out = []

        for size, stride in self.patch_size_stride:
            o = self.to_patches(x, size=size, stride=stride)
            out.append(o)

        return out

    def to_patches(self, image, size=64, stride=8):
        b, s = image.shape[:2]

        patches = (
            image.unfold(1, s, stride).unfold(2, size, stride).unfold(3, size, stride)
        )  # [bs, 1, num_height, num_width, dim, size, size]

        h, w = patches.shape[2:4]
        patches = patches.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        return patches


class PatchContrastiveNCE(BaseModule):
    def __init__(
        self,
        loss_weight=1.0,
        nce_t=0.07,
        gradient_flows_to_negative_nce=False,
        nce_fake_negatives=True,
        nc=256,
        num_feats=3,
        input_size=8,
        num_patches=256,
        layer_weight_multiplier=1.0,
    ) -> None:
        # input_nc = 4 in latent space
        super().__init__(loss_weight=loss_weight)
        self.encoder = LatentEncoder(input_size)
        self.projector = PatchLatentProjector(nc=nc, input_ncs=self.encoder.output_channels, num_feats=num_feats)
        self.patchnceloss = PatchNCELoss(
            nce_t=nce_t,
            gradient_flows_to_negative_nce=gradient_flows_to_negative_nce,
            nce_fake_negatives=nce_fake_negatives,
        )
        self.num_patches = num_patches
        layer_weights = np.array([layer_weight_multiplier**i for i in range(num_feats)])
        self.layer_weights = layer_weights / layer_weights.sum()
        self.num_feats = num_feats
        self.input_size = input_size

    def forward(self, target, reconstruct):
        target_feats = self.encoder(target)
        recons_feats = self.encoder(reconstruct)

        target_prjs, ids = self.projector(
            target_feats, num_patches=self.num_patches, patch_ids=None
        )  # shape [B,N,D]xnum_feats
        recons_prjs, _ = self.projector(recons_feats, num_patches=self.num_patches, patch_ids=ids)  # shape B,N,D

        total_loss = 0
        for target_prj, recons_prj, layer_weight in zip(target_prjs, recons_prjs, self.layer_weights):
            loss = self.patchnceloss(recons_prj, target_prj) * layer_weight
            loss += self.patchnceloss(target_prj, recons_prj) * layer_weight
            total_loss += loss.mean()

        return self._ouput_loss(total_loss / self.num_feats)

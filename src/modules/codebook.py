import torch
import torch.nn as nn
from torch.nn import functional as F


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors=128, latent_dim=512, beta=0.25, downsample=1):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        assert latent_dim % downsample == 0
        project_dim = int(latent_dim // downsample)
        self.project_dim = project_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, project_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        if downsample > 1:
            self.quant_conv = nn.Conv2d(latent_dim, project_dim, 1)
            self.post_quant_conv = nn.Conv2d(project_dim, latent_dim, 1)

    def compute_codebook(self, z):
        z = F.normalize(z, p=2, dim=-1)
        z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(-1, self.project_dim)

        weight = self.embedding.weight
        weight = F.normalize(weight, p=2, dim=-1)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(weight**2, dim=1)
            - 2 * (torch.matmul(z_flattened, weight.t()))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if not self.training:
            z_q = z_q.permute(0, 3, 1, 2)
            return z_q, min_encoding_indices

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # Straight Through Estimator
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss

    def forward(self, z):
        if hasattr(self, "quant_conv"):
            z = self.quant_conv(z)
        if self.training:
            z_q, min_encoding_indices, loss = self.compute_codebook(z)
        else:
            z_q, min_encoding_indices = self.compute_codebook(z)
        if hasattr(self, "post_quant_conv"):
            z_q = self.post_quant_conv(z_q)
        if self.training:
            return z_q, min_encoding_indices, loss
        else:
            return z_q, min_encoding_indices

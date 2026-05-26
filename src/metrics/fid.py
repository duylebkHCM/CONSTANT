import os
import numpy as np
from PIL import Image
from scipy import linalg
import torch
import torchvision.transforms as TF
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().absolute().parent))
from .inception import InceptionV3


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def collate_fn(self, batch):
        imgs_width = [x.size(2) for x in batch]
        max_width = max(imgs_width)
        imgs = torch.stack(
            [torch.nn.functional.pad(x, (0, max_width - x.size(2))) for x in batch],
            dim=0,
        )
        return imgs


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class FIDScore:
    def __init__(self, batch_size, dims, device="cuda"):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = 0
        self.device = device

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ("fid calculation produces singular product; adding %s to diagonal of cov estimates") % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def get_activations(self, datasource):
        """Calculates the activations of the pool_3 layer for all images.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        if self.batch_size > len(datasource):
            print(("Warning: batch size is bigger than the data size. Setting batch size to data size"))
            self.batch_size = len(datasource)

        if self.batch_size > 1:
            dataset = ImagePathDataset(datasource, transforms=TF.ToTensor())
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=self.num_workers,
                pin_memory=False,
            )
        else:
            dataset = ImagePathDataset(datasource, transforms=TF.ToTensor())
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )

        pred_arr = np.empty((len(datasource), self.dims))

        start_idx = 0
        for idx, batch in tqdm(enumerate(dataloader)):
            images = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(images)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        return pred_arr

    def calculate_activation_statistics(self, datasource):
        """Calculation of the statistics used by the FID
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the inception model.
        """
        act = self.get_activations(datasource)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def compute_statistics_of_datasource(self, datasource):
        if isinstance(datasource, str) and datasource.endswith(".npz"):
            with np.load(datasource) as f:
                m, s = f["mu"][:], f["sigma"][:]
        elif isinstance(datasource, str) and pathlib.Path(datasource).is_dir:
            path = pathlib.Path(datasource)
            files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.rglob("*.{}".format(ext))])
            m, s = self.calculate_activation_statistics(files)
        return m, s

    def compute_scores(self, real_path, fake_path):
        """Calculates the FID of two paths"""
        for p in [real_path, fake_path]:
            if not os.path.exists(p):
                raise RuntimeError("Invalid path: %s" % p)

        m1, s1 = self.compute_statistics_of_datasource(real_path)
        m2, s2 = self.compute_statistics_of_datasource(fake_path)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value

import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from ..modules.utils import default
from ..modules.unet import UNetModel
from .base import BaseModule
from ..data.constant import CHAR_WIDTH


class DiffusionEngine(BaseModule):
    def __init__(
        self,
        backbone_params={},
        clfree_params={},
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        input_channel=3,
        latent=True,
        ddim_sampling=True,
        ddim_eta=0.0,
        sampling_steps=30,
        fp16=False,
        down_sample=8,
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1.0,
        loss_weight=1.0,
    ):
        super().__init__(loss_weight=loss_weight)

        self.input_channel = input_channel
        self.origin_img_size = img_size
        self.img_size = int(img_size // down_sample)
        self.down_sample = down_sample

        self.clfree_params = clfree_params
        self.latent = latent
        self.pixel_space = not latent
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        betas = self.prepare_noise_schedule()
        register_buffer("betas", betas)
        register_buffer("alphas", 1.0 - betas)
        register_buffer("alphas_cumprod", torch.cumprod(self.alphas, axis=0))
        register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_inv_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
        register_buffer(
            "sqrt_inv_alphas_cumprod_minus_one",
            torch.sqrt(1.0 / self.alphas_cumprod - 1),
        )

        self.model = self.create_models(backbone_params)
        self.fp16 = fp16
        if fp16:
            self.model.convert_to_fp16()

        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value

        self.use_ddim_sampling = ddim_sampling
        self.ddim_eta = ddim_eta
        self.sampling_steps = default(sampling_steps, noise_steps) if ddim_sampling else noise_steps
        assert self.sampling_steps <= noise_steps

    @property
    def device(self):
        return next(self.model.parameters()).device

    def update_sampling_params(self, guidance_scale=None, sampling_steps=None):
        """Update classifier-free guidance and sampling steps parameters on the fly."""
        if guidance_scale is not None:
            self.clfree_params["cond_scale"] = guidance_scale
        if sampling_steps is not None:
            if sampling_steps > self.noise_steps:
                raise ValueError(f"sampling_steps ({sampling_steps}) cannot exceed noise_steps ({self.noise_steps})")
            self.sampling_steps = int(sampling_steps)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def create_models(self, backbone_params):
        backbone_params["in_channels"] = backbone_params["out_channels"] = self.input_channel
        unet = UNetModel(**backbone_params)
        return unet

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.extract(self.alphas_cumprod, t, x.shape))
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.extract(self.alphas_cumprod, t, x.shape))
        phi = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * phi, phi

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,)).to(self.device)

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Taken from diffusers
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing pixels from saturation at each step. We find that dynamic thresholding results in significantly better photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def forward(self, vae, x_start, text_features, style_features):
        if self.latent:
            x_start = vae.encode(x_start.to(torch.float32)).latent_dist.sample()
            x_start = x_start * 0.18215

        batch_size = x_start.shape[0]

        t = self.sample_timesteps(batch_size)
        x_t, noise = self.noise_images(x_start, t)

        pred_noise = self.model(
            x_t,
            timesteps=t,
            style_features=style_features,
            text_features=text_features,
            cond_drop_prob=self.clfree_params.get("cond_drop_prob", 0.0),
        )
        if self.fp16:
            pred_noise = pred_noise.float()

        loss = F.mse_loss(pred_noise, noise)

        return {
            "pred_noise": pred_noise,
            "noise": noise,
            "timestep": t,
            "x_t": x_t,
            "x_start": x_start,
            "denoise_loss": self._ouput_loss(loss),
        }

    def _predict_x_start_from_noise(self, epsilon, t, x_t):
        sqrt_inv_alphas_cumprod_t = self.extract(self.sqrt_inv_alphas_cumprod, t, x_t.shape)
        sqrt_inv_alphas_cumprod_minus_one_t = self.extract(self.sqrt_inv_alphas_cumprod_minus_one, t, x_t.shape)
        x_start = sqrt_inv_alphas_cumprod_t * x_t - sqrt_inv_alphas_cumprod_minus_one_t * epsilon
        # Only use when perform sampling on pixel space
        if self.pixel_space:
            if self.thresholding:
                x_start = self._threshold_sample(x_start)
            else:
                x_start = torch.clamp(x_start, min=-1.0, max=1.0)

        return x_start

    @torch.inference_mode()
    def ddim_sampling(self, x_t, n, text_features, style_features, log_progress=False):
        time_steps = torch.linspace(-1, self.noise_steps - 1, steps=self.sampling_steps + 1)
        time_steps = list(reversed(time_steps.int().tolist()))
        time_pairs = list(zip(time_steps[:-1], time_steps[1:]))

        denoise_lst = []

        for time, time_prev in tqdm(time_pairs, position=0):
            t = (torch.ones(n) * time).long().to(self.device)

            predicted_noise = self.model.forward_sampling(
                x=x_t,
                timesteps=t,
                style_features=style_features,
                text_features=text_features,
                **self.clfree_params,
            )
            if self.fp16:
                predicted_noise = predicted_noise.float()

            x_start = self._predict_x_start_from_noise(predicted_noise, t, x_t)
            sqrt_inv_alphas_cumprod_t = self.extract(self.sqrt_inv_alphas_cumprod, t, x_t.shape)
            sqrt_inv_alphas_cumprod_minus_one_t = self.extract(self.sqrt_inv_alphas_cumprod_minus_one, t, x_t.shape)
            predicted_noise = (sqrt_inv_alphas_cumprod_t * x_t - x_start) / sqrt_inv_alphas_cumprod_minus_one_t

            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev]

            # From equation 16 of paper DDIM
            # If ddim_eta = 0 we have a deterministic sampling process
            sigma = self.ddim_eta * ((1 - alpha / alpha_prev) * (1 - alpha_prev) / (1 - alpha)).sqrt()
            c = (1 - alpha_prev - sigma**2).sqrt()

            noise = torch.randn_like(x_t)

            x_t = x_start * torch.sqrt(alpha_prev) + c * predicted_noise + sigma * noise
            denoise_lst.append(x_t.unsqueeze(dim=1))

        if log_progress:
            denoise_lst.append(x_start.unsqueeze(dim=1))
            return x_start, denoise_lst
        else:
            return x_start

    def ddim_with_grad_sampling(
        self,
        vae,
        original_lens,
        text_features,
        style_features,
        guidance_func=None,
        guidance_labels=None,
    ):
        n = len(style_features)

        x_t = torch.randn(
            (
                n,
                self.input_channel,
                self.img_size,
                int(text_features.shape[1] * CHAR_WIDTH // self.down_sample),
            )
        ).to(self.device)

        time_steps = torch.linspace(-1, self.noise_steps - 1, steps=self.sampling_steps + 1)
        time_steps = list(reversed(time_steps.int().tolist()))
        time_pairs = list(zip(time_steps[:-1], time_steps[1:]))

        num_steps = guidance_func.num_optim_steps
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        x_start = x_t
        for time, time_prev in tqdm(time_pairs, position=0):
            t = (torch.ones(n) * time).long().to(self.device)

            if time_prev < 0:
                x0 = self.sampling_postprocess(vae, x_start)
                x0 = torch.unbind(x0, dim=0)
                continue

            for _ in range(num_steps):
                torch.set_grad_enabled(True)

                img_in = x_t.detach().requires_grad_(True)

                predicted_noise = self.model.forward_sampling(
                    x=img_in,
                    timesteps=t,
                    style_features=style_features,
                    text_features=text_features,
                    trigger_style_attn=False,
                    **self.clfree_params,
                )
                if self.fp16:
                    predicted_noise = predicted_noise.float()

                x_start = self._predict_x_start_from_noise(predicted_noise, t, img_in)
                latents = 1 / 0.18215 * x_start
                recons_images = vae.decode(latents).sample
                recons_images = (recons_images / 2 + 0.5).clamp(0, 1) * 255.0
                selected = -1 * guidance_func.cal_loss(recons_images, guidance_labels)

                grad = torch.autograd.grad(selected, img_in)[0]
                grad = grad * guidance_func.optim_guidance_3_wt

                predicted_noise = (
                    predicted_noise - self.extract(sqrt_one_minus_alpha_cumprod, t, img_in.shape) * grad.detach()
                )
                img_in = img_in.requires_grad_(False)

                del img_in, x_start, recons_images, selected, grad

                torch.set_grad_enabled(False)

                with torch.no_grad():
                    x_start = self._predict_x_start_from_noise(predicted_noise, t, x_t)

                    sqrt_inv_alphas_cumprod_t = self.extract(self.sqrt_inv_alphas_cumprod, t, x_t.shape)
                    sqrt_inv_alphas_cumprod_minus_one_t = self.extract(
                        self.sqrt_inv_alphas_cumprod_minus_one, t, x_t.shape
                    )
                    predicted_noise = (sqrt_inv_alphas_cumprod_t * x_t - x_start) / sqrt_inv_alphas_cumprod_minus_one_t

                    alpha = self.alphas_cumprod[time]
                    alpha_prev = self.alphas_cumprod[time_prev]

                    # From equation 16 of paper DDIM
                    # If ddim_eta = 0 we have a deterministic sampling process
                    sigma = self.ddim_eta * ((1 - alpha / alpha_prev) * (1 - alpha_prev) / (1 - alpha)).sqrt()
                    c = (1 - alpha_prev - sigma**2).sqrt()

                    noise = torch.randn_like(x_t)

                    x_prev = x_start * torch.sqrt(alpha_prev) + c * predicted_noise + sigma * noise
                    x_t = torch.sqrt(alpha / alpha_prev) * x_prev + torch.sqrt(
                        1 - alpha / alpha_prev
                    ) * torch.rand_like(x_prev)
                    del sigma, c

            x_t = x_prev

        postprocess_x = []
        for origin_len, img in zip(original_lens, x0):
            postprocess_x.append(img[:, :, :origin_len])
        return postprocess_x

    @torch.inference_mode()
    def ddpm_sampling(self, x_t, n, text_features, style_features, log_progress=False):
        denoise_lst = []

        for i in tqdm(list(reversed(range(0, self.sampling_steps))), position=0):
            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = self.model.forward_sampling(
                x=x_t,
                timesteps=t,
                style_features=style_features,
                text_features=text_features,
                **self.clfree_params,
            )
            if self.fp16:
                predicted_noise = predicted_noise.float()

            alpha = self.extract(self.alphas, t, x_t.shape)
            alpha_hat = self.extract(self.alphas_cumprod, t, x_t.shape)
            beta = self.extract(self.betas, t, x_t.shape)

            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            x_t = (
                1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                + torch.sqrt(beta) * noise
            )
            denoise_lst.append(x_t.unsqueeze(dim=1))

        if log_progress:
            return x_t, denoise_lst
        else:
            return x_t

    def sampling_postprocess(self, vae, x, visualize=True):
        if self.latent:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            if visualize:
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = torch.from_numpy(image)
                image = image.permute(0, 3, 1, 2)
        else:
            image = (x.clamp(-1, 1) + 1) / 2
            if visualize:
                image = (image * 255).type(torch.uint8)

        return image

    @torch.inference_mode()
    def sampling(
        self,
        vae,
        text_features,
        style_features,
        input_shape: tuple,
        original_lens=None,
        log_progress=False,
    ):
        n = len(style_features)

        x = torch.randn((n, self.input_channel, *input_shape)).to(self.device)

        if self.use_ddim_sampling:
            result = self.ddim_sampling(x, n, text_features, style_features, log_progress)
        else:
            result = self.ddpm_sampling(x, n, text_features, style_features, log_progress)

        if log_progress:
            return result
        else:
            x = self.sampling_postprocess(vae, result)
            unbinded = torch.unbind(x, dim=0)
            if original_lens:
                postprocess_x = []
                for origin_len, img in zip(original_lens, unbinded):
                    postprocess_x.append(img[:, :, :origin_len])
                return postprocess_x
            return unbinded

import copy
from abc import abstractmethod
from torch import nn
import torch
from torchvision.utils import save_image
from typing import Optional
from omegaconf import OmegaConf
from itertools import chain
from typing import Union, Tuple, Any, Dict, List
from torch import optim
from functools import partial
from contextlib import suppress
from diffusers import AutoencoderKL
from .base import Loss_Storage
from ..data.constant import CHAR_WIDTH
from ..modules.ema import EMA
from ..modules.scheduler import get_scheduler
from ..modules.helpers import NativeScaler, NoneScaler
from ..helpers import words_to_images, initialize_from_config, resolve_pretrained_path


class BasePipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vae: Any = None
        self.diffusion: Any = None
        self.style_extractor: Any = None
        self.text_encoder: Any = None
        self.main_module_names: Any = {}
        self.optimizer: Any = None

    def save_state_dict(self, save_path):
        state_dict = dict(map(lambda kv: (kv, getattr(self, kv).state_dict()), self.main_module_names))
        torch.save(state_dict, save_path)

    def save_state_dict_ema(self, save_path):
        assert hasattr(self, "main_modules")
        state_dict = dict(map(lambda kv: (kv[0], kv[1].state_dict()), self.main_modules.items()))
        torch.save(state_dict, save_path)

    def save_checkpoint(self, save_path, cur_iteration, best_loss=None, cur_epoch=None):
        save_modules = {}
        for name, module in self.named_children():
            if name == "vae":
                continue
            save_modules[name] = module.state_dict()

        save_ckpt = {
            "model": save_modules,
            "optmizer": self.optimizer.state_dict(),
            "cur_iter": cur_iteration,
        }
        if cur_epoch is not None:
            save_ckpt["cur_epoch"] = cur_epoch
        if best_loss is not None:
            save_ckpt["best_loss"] = best_loss

        torch.save(save_ckpt, save_path)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cuda")
        for module_name, state_dict in ckpt["model"].items():
            getattr(self, module_name).load_state_dict(state_dict)
        if ckpt.get("optmizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optmizer"])
        return (
            ckpt["cur_iter"],
            ckpt.get("cur_epoch", None),
            ckpt.get("best_loss", None),
        )

    @abstractmethod
    def get_conditioning(self, style_images, text_embeddings):  # type: ignore[no-untyped-def]
        raise NotImplementedError()

    @property
    def _style_extractor(self):
        return (
            self.main_modules["style_extractor"]
            if hasattr(self, "main_modules") and not self.training
            else self.style_extractor
        )

    @property
    def _text_encoder(self):
        return (
            self.main_modules["text_encoder"]
            if hasattr(self, "main_modules") and not self.training
            else self.text_encoder
        )

    @property
    def _diffusion(self):
        return self.main_modules["diffusion"] if hasattr(self, "main_modules") and not self.training else self.diffusion

    @torch.inference_mode()
    def sampling(
        self,
        input_shape,
        original_lens,
        text_embeddings,
        style_images,
        log_progress,
        guidance_scale=None,
        ddim_steps=None,
    ) -> Union[Tuple[Any], torch.Tensor]:
        if guidance_scale is not None or ddim_steps is not None:
            self._diffusion.update_sampling_params(guidance_scale=guidance_scale, sampling_steps=ddim_steps)

        style_features, text_features = self.get_conditioning(style_images, text_embeddings)
        sampling_results = self._diffusion.sampling(
            self.vae,
            text_features,
            style_features,
            input_shape,
            original_lens,
            log_progress=log_progress,
        )
        return sampling_results

    def sampling_with_grad(
        self,
        original_lens,
        text_embeddings,
        style_images,
        guidance_func,
        guidance_labels,
    ) -> Union[Tuple[Any], torch.Tensor]:
        torch.set_grad_enabled(False)
        style_features, text_features = self.get_conditioning(style_images, text_embeddings)
        sampling_results = self._diffusion.ddim_with_grad_sampling(
            self.vae,
            original_lens,
            text_features,
            style_features,
            guidance_func=guidance_func,
            guidance_labels=guidance_labels,
        )
        return sampling_results

    @torch.inference_mode()
    def log_images(self, batch, save_path=None, num_vis_steps=10):
        NUM_SAMPLES = 4
        text_embeddings, style_images, raw_labels, original_lens = (
            batch["text_embeddings"],
            batch["style_images"],
            batch["raw_labels"],
            batch["original_lens"],
        )
        input_shape = (
            self._diffusion.img_size,
            int(text_embeddings.shape[1] * CHAR_WIDTH // self._diffusion.down_sample),
        )
        denoise_lst = self.sampling(input_shape, original_lens, text_embeddings, style_images, log_progress=True)[-1]
        denoise_lst = torch.cat(denoise_lst, dim=1)
        selected_indices = [
            idx
            for idx in range(
                self.diffusion.sampling_steps - 1,
                0,
                -self.diffusion.sampling_steps // num_vis_steps,
            )
        ]
        selected_indices = torch.tensor(list(reversed(selected_indices)), device=denoise_lst.device)
        denoise_lst = torch.index_select(denoise_lst, dim=1, index=selected_indices)
        denoise_lst = denoise_lst[:NUM_SAMPLES]
        denoise_lst = denoise_lst.permute(1, 0, 2, 3, 4)
        new_denoise_lst = []
        for denoise_img in denoise_lst:
            denoise_img = self.diffusion.sampling_postprocess(self.vae, denoise_img)
            new_denoise_lst.append(denoise_img)
        new_denoise_lst = torch.stack(new_denoise_lst, dim=0).permute(1, 0, 2, 3, 4)

        # Concat with style images and rendered text for visually comparison
        rendered_texts = words_to_images(
            raw_labels,
            int(self._diffusion.img_size * self._diffusion.down_sample),
            int(text_embeddings.shape[1] * CHAR_WIDTH),
            n_channel=3,
        )[:NUM_SAMPLES]

        final_denoise_lst = torch.concat(
            [
                new_denoise_lst,
                rendered_texts.unsqueeze(1),
                style_images.unsqueeze(dim=1).cpu()[:NUM_SAMPLES],
            ],
            dim=1,
        )

        # Concat target images to compare with
        target_images = batch["images"]
        final_denoise_lst = torch.concat(
            [final_denoise_lst, target_images.unsqueeze(dim=1).cpu()[:NUM_SAMPLES]],
            dim=1,
        )
        final_denoise_lst = final_denoise_lst.flatten(start_dim=0, end_dim=1)
        save_image(final_denoise_lst, save_path, nrow=num_vis_steps + 3)

    @torch.inference_mode()
    def image_generator(self, dataloader):
        """Use as generator for validation step"""
        for batch in dataloader:
            text_embeddings = batch["text_embeddings"].to(self.device)
            style_images = batch["style_images"].to(self.device)
            gen_images = self.sampling(text_embeddings, style_images, log_progress=False)
            yield {"images": gen_images}


class HandwritingGenerationPipeline(BasePipeline):
    def __init__(
        self,
        vae_config,
        diffusion_config,
        style_extractor_config,
        text_encoder_config,
        style_constrastive_config: Optional[OmegaConf] = None,
        content_constrastive_config: Optional[OmegaConf] = None,
        loss_balancing=False,
        grad_clipping=False,
    ):
        super().__init__()
        self.ema = None
        self.autocast = suppress
        self.scaler = NoneScaler()
        self.loss_balancing = loss_balancing
        self.grad_clipping = grad_clipping

        if diffusion_config.params.latent:
            vae_cache_dir = resolve_pretrained_path(vae_config)
            self.vae = AutoencoderKL.from_pretrained(vae_cache_dir)
            self.vae.eval().requires_grad_(False)
        else:
            self.vae = nn.Identity()
        self.diffusion = initialize_from_config(diffusion_config)
        self.style_extractor = initialize_from_config(style_extractor_config)
        self.text_encoder = initialize_from_config(text_encoder_config)

        # Training only supported modules
        if style_constrastive_config is not None:
            self.style_constrastive_enchance = initialize_from_config(style_constrastive_config)
        if content_constrastive_config is not None:
            self.patch_constrastive_enchance = initialize_from_config(content_constrastive_config)

        self.main_module_names: List[str] = ["diffusion", "style_extractor", "text_encoder"]

    def train(self):
        self.diffusion.train()
        self.style_extractor.train()
        self.text_encoder.train()
        if hasattr(self, "style_constrastive_enchance"):
            self.style_constrastive_enchance.train()
        if hasattr(self, "patch_constrastive_enchance"):
            self.patch_constrastive_enchance.train()

    def eval(self):
        self.diffusion.eval()
        self.style_extractor.eval()
        self.text_encoder.eval()

    def configure_optimizers(self, lr: Union[float, Dict[str, float]]):
        self.diffusion_params = chain(
            self.diffusion.parameters(),
            self.text_encoder.parameters(),
            self.style_extractor.parameters(),
            self.style_constrastive_enchance.parameters() if hasattr(self, "style_constrastive_enchance") else [],  # type: ignore[attr-defined]
            self.patch_constrastive_enchance.parameters() if hasattr(self, "patch_constrastive_enchance") else [],  # type: ignore[attr-defined]
        )
        self.optimizer = optim.AdamW(self.diffusion_params, lr)  # type: ignore[arg-type]

    def setup_ema(self):
        self.ema = EMA(0.999)
        self.main_modules = dict(
            map(
                lambda name: (name, copy.deepcopy(getattr(self, name))),
                self.main_module_names,
            )
        )

    def setup_amp(self):
        self.autocast = partial(torch.autocast, device_type=self.device.type, dtype=torch.float16)
        self.scaler = NativeScaler()

    def setup_scheduler(self, scheduler_config: OmegaConf, num_training_steps: int):
        self.scheduler = {}
        scheduler_config = OmegaConf.to_object(scheduler_config)
        name = scheduler_config.pop("type")
        self.scheduler = get_scheduler(
            name,
            self.optimizer,
            **scheduler_config,
            num_training_steps=num_training_steps,
        )

    def step_ema(self):
        assert hasattr(self, "ema")
        for module_name, module in self.main_modules.items():
            self.ema.step_ema(module, getattr(self, module_name))

    def save_state_dict(self, save_path):
        state_dict = dict(map(lambda kv: (kv, getattr(self, kv).state_dict()), self.main_module_names))
        torch.save(state_dict, save_path)

    def save_state_dict_ema(self, save_path):
        assert hasattr(self, "main_modules")
        state_dict = dict(map(lambda kv: (kv[0], kv[1].state_dict()), self.main_modules.items()))
        torch.save(state_dict, save_path)

    def load_state_dict(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        for k, state_dict in ckpt.items():
            getattr(self, k).load_state_dict(state_dict)

    @property
    def device(self):
        return next(self.diffusion.parameters()).device

    def get_conditioning(self, style_images, text_embeddings):
        style_features = self._style_extractor(style_images)
        text_features = self._text_encoder(text_embeddings)
        return style_features, text_features

    def forward(self, images, anchor_images, style_images, text_embeddings, **kwargs):
        anchor_images = (anchor_images + 1) * 0.5

        loss_combinations = {}

        text_features = self._text_encoder(text_embeddings)
        if hasattr(self, "style_constrastive_enchance"):
            style_features, style_seqs, vq_loss = self._style_extractor(style_images)
            target_features, *_ = self._style_extractor(anchor_images)
            style_loss = self.style_constrastive_enchance(target_features, style_features)
            loss_combinations["style_loss"] = style_loss
        else:
            style_features, style_seqs, vq_loss = self._style_extractor(style_images)

        loss_combinations["vq_loss"] = vq_loss

        result = self.diffusion(self.vae, images, text_features, style_seqs)

        loss_combinations["denoise_loss"] = result["denoise_loss"]

        if hasattr(self, "patch_constrastive_enchance"):
            reconstruct_x_start_t = self.diffusion._predict_x_start_from_noise(
                result["pred_noise"], result["timestep"], result["x_t"]
            )
            perceptual_loss = self.patch_constrastive_enchance(result["x_start"], reconstruct_x_start_t)
            loss_combinations["perceptual_loss"] = perceptual_loss

        if self.loss_balancing:
            grad_combination = {}
            if hasattr(self, "patch_constrastive_enchance"):
                denoise_grad_1 = torch.autograd.grad(
                    result["denoise_loss"].loss_value,
                    result["pred_noise"],
                    create_graph=True,
                    retain_graph=True,
                )[0]
                perceptual_grad = torch.autograd.grad(
                    perceptual_loss.loss_value,
                    reconstruct_x_start_t,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                perceptual_weight = (
                    self.patch_constrastive_enchance.loss_weight
                    * torch.div(torch.std(denoise_grad_1), 1e-7 + torch.std(perceptual_grad)).detach()
                )
                grad_combination["perceptual_loss"] = perceptual_weight

            if hasattr(self, "style_constrastive_enchance"):
                denoise_grad_2 = torch.autograd.grad(
                    result["denoise_loss"].loss_value,
                    style_seqs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                style_grad = torch.autograd.grad(
                    style_loss.loss_value,
                    style_features,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                style_grad2 = torch.autograd.grad(
                    style_loss.loss_value,
                    target_features,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                style_weight = (
                    self.style_constrastive_enchance.loss_weight
                    * torch.div(torch.std(denoise_grad_2), 1e-7 + torch.std(style_grad)).detach()
                )
                style_weight2 = (
                    self.style_constrastive_enchance.loss_weight
                    * torch.div(torch.std(denoise_grad_2), 1e-7 + torch.std(style_grad2)).detach()
                )
                grad_combination["style_loss"] = (style_weight + style_weight2) / 2

            return loss_combinations, grad_combination
        else:
            return loss_combinations

    @torch.inference_mode()
    def validation_step(self, batch):
        style_images, text_embeddings, target_images = (
            batch["style_images"],
            batch["text_embeddings"],
            batch["images"],
        )
        style_features, text_features = self.get_conditioning(style_images, text_embeddings)
        result = self.diffusion(self.vae, target_images, text_features, style_features)
        return result["denoise_loss"].loss_value

    def _combine_loss(
        self,
        combine_loss_keyword,
        loss_combination: Dict[str, Loss_Storage],
        grad_combination: Dict[str, torch.Tensor],
    ):
        if self.loss_balancing and len(grad_combination) > 0:
            final_loss = sum(
                [
                    loss_info.loss_value
                    * (
                        grad_combination.get(loss_name, None)
                        if grad_combination.get(loss_name, None) is not None
                        else loss_info.loss_weight
                    )
                    for loss_name, loss_info in loss_combination.items()
                ]
            )
        else:
            final_loss = sum([loss_info.loss_value * loss_info.loss_weight for loss_info in loss_combination.values()])

        return {
            combine_loss_keyword: final_loss,
            **{keyword: loss_info.loss_value for keyword, loss_info in loss_combination.items()},
        }

    def forward_backward(self, *args, **kwargs):
        self.optimizer.zero_grad()

        with self.autocast():
            grad_combine = {}
            if self.loss_balancing:
                loss_combine, grad_combine = self(*args, **kwargs)
            else:
                loss_combine = self(*args, **kwargs)
            loss_dict = self._combine_loss("total_loss", loss_combine, grad_combine)

        if torch.any(torch.isnan(loss_dict["total_loss"])).item():
            print("Exists NaN value in current batch loss, skip this update")
            return None

        if self.grad_clipping:
            self.scaler(
                loss_dict["total_loss"],
                self.optimizer,
                self.diffusion_params,
                self.grad_clipping,
            )
        else:
            self.scaler(loss_dict["total_loss"], self.optimizer)

        if hasattr(self, "scheduler"):
            self.scheduler.step()

        return loss_dict

    def forward_backward_update(self, *args, **kwargs):
        return self.forward_backward(*args, **kwargs)

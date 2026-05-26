import os
from pathlib import Path
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
import copy
import time
from collections import defaultdict
from typing import Dict, Union, List
from loguru import logger
from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter
from helpers import (
    initialize_from_config,
    set_seed,
    count_parameters,
    get_obj_from_str,
    make_dirs,
    cal_elasped_time,
    override_config,
    determine_optimal_num_workers,
)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Not using distributed training")
        return 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    dist.barrier()

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is main process (rank 0)."""
    return rank == 0


class Trainer:
    def __init__(
        self,
        device,
        config_path,
        ckpt_path=None,
        pretrained_path=None,
        config_overrides=None,
        rank=0,
        world_size=1,
        local_rank=0,
        use_ddp=False,
    ):
        """
        Initialize trainer.

        Args:
            device: Device to use for training
            config_path: Path to config file
            ckpt_path: Path to checkpoint for resuming training
            pretrained_path: Path to pretrained weights
            config_overrides: Config overrides
            rank: Process rank for DDP
            world_size: Total number of processes for DDP
            local_rank: Local GPU rank for DDP
            use_ddp: Whether to use DistributedDataParallel
        """
        config = OmegaConf.load(config_path)

        if config_overrides:
            config = override_config(config, config_overrides)

        self.config = config
        self.config_path = config_path
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.use_ddp = use_ddp
        self.is_main = is_main_process(rank) if use_ddp else True

        num_workers = getattr(config.training, "num_workers", None)
        if num_workers is None or num_workers == "auto":
            num_workers = determine_optimal_num_workers(
                f"cuda:{local_rank}" if use_ddp else device, verbose=self.is_main
            )
            OmegaConf.update(config.training, "num_workers", num_workers)
            if self.is_main:
                logger.info(f"Auto-configured num_workers to {num_workers}")
        else:
            if self.is_main:
                logger.info(f"Using configured num_workers: {num_workers}")

        dataset = initialize_from_config(self.config.dataset)
        self.dataset = dataset

        # Handle dataloader setup for both DDP and non-DDP
        if use_ddp:
            # Use DistributedSampler for DDP training
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

            # Adjust batch size to be per-GPU batch size
            if hasattr(self.config.training, "batch_size"):
                batch_size = self.config.training.batch_size // world_size
            else:
                batch_size = 1

            self.loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self.sampler,
                pin_memory=True,
                collate_fn=dataset.collate_function,
                num_workers=self.config.training.num_workers,
            )

            # Set epoch for sampler to ensure different shuffling each epoch
            self.sampler.set_epoch(0)
        else:
            if hasattr(self.config.training, "sampler"):
                self.loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_sampler=get_obj_from_str(self.config.training.sampler.target)(
                        dataset,
                        shuffle=True,
                        batch_size=self.config.training.batch_size,
                        drop_last=True,
                    ),
                    collate_fn=dataset.collate_function,
                    pin_memory=True,
                    num_workers=self.config.training.num_workers,
                )
            else:
                self.loader = torch.utils.data.DataLoader(
                    dataset,
                    self.config.training.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    collate_fn=dataset.collate_function,
                    num_workers=self.config.training.num_workers,
                )
            self.sampler = None

        # Get a fixed batch for visualization (only on main process)
        if self.is_main:
            self.batch_images = copy.deepcopy(next(iter(self.loader)))
        else:
            self.batch_images = None

        # Setup validation dataloader
        if hasattr(self.config, "testdataset") and self.config.training.valid:
            if self.is_main:
                test_dataset = initialize_from_config(self.config.testdataset)
                self.test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    self.config.training.val_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    collate_fn=dataset.collate_function,
                    num_workers=self.config.training.num_workers,
                )
                self.top_k_track = []
                self.max_top_k = self.config.metric.save_top_k
            else:
                self.test_loader = None
                self.top_k_track = None
                self.max_top_k = None
        else:
            self.test_loader = None

        self.config.text_encoder.params.input_size = dataset.tokenizer.vocab_size

        # Initialize pipeline
        pipeline = get_obj_from_str(self.config.pipeline.target)(
            vae_config=self.config.vae.pretrained_path,
            diffusion_config=self.config.diffusion,
            style_extractor_config=self.config.style_extractor,
            text_encoder_config=self.config.text_encoder,
            style_constrastive_config=self.config.constrastive,
            content_constrastive_config=self.config.content_constrastive,
            loss_balancing=self.config.training.loss_balancing,
        )
        pipeline.configure_optimizers(self.config.training.base_lr)

        # Move to appropriate device
        if use_ddp:
            pipeline = pipeline.to(local_rank)
            # Wrap model with DDP
            self.pipeline = DDP(
                pipeline,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
            device_obj = torch.device(f"cuda:{local_rank}")
        else:
            pipeline = pipeline.to(device)
            self.pipeline = pipeline
            device_obj = device

        self.cur_iter = None
        self.cur_epoch = None
        self.best_loss = float("inf")
        self.continue_training = False

        if ckpt_path is not None:
            if use_ddp:
                cur_iter, _, best_loss = self.pipeline.module.load_checkpoint(ckpt_path)
            else:
                cur_iter, _, best_loss = pipeline.load_checkpoint(ckpt_path)
            self.cur_iter = cur_iter
            if best_loss is not None:
                self.best_loss = best_loss
            self.continue_training = True

        if pretrained_path is not None:
            if use_ddp:
                self.pipeline.module.load_state_dict(pretrained_path)
            else:
                pipeline.load_state_dict(pretrained_path)

        # Setup EMA and AMP
        if self.config.training.update_ema:
            if use_ddp:
                self.pipeline.module.setup_ema()
            else:
                self.pipeline.setup_ema()

        if self.config.training.amp:
            if use_ddp:
                self.pipeline.module.setup_amp()
            else:
                self.pipeline.setup_amp()

        if hasattr(self.config.training, "scheduler"):
            if use_ddp:
                self.pipeline.module.setup_scheduler(
                    self.config.training.scheduler,
                    num_training_steps=self.config.training.iteration,
                )
            else:
                self.pipeline.setup_scheduler(
                    self.config.training.scheduler,
                    num_training_steps=self.config.training.iteration,
                )

        self.total_iterations = self.config.training.iteration
        self.device = device_obj

    def setup_exp(self):
        if not self.is_main:
            return

        timestamp = int(round(time.time() * 1000))
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp / 1000))
        # Setup experiment log dir
        config_rel_path = (
            Path(self.config_path)
            .absolute()
            .relative_to(Path(__file__).absolute().parent.parent.joinpath("configs"))
            .parent.joinpath(Path(self.config_path).stem)
        )
        # output of each experiment follow the relative structure of the config file used for training with the configs directory, this aim for easy to track the list of experiments
        exp_dir_suffix = f"{timestamp}_ddp_{self.world_size}gpu" if self.use_ddp else timestamp
        exp_dir = Path(self.config.training.output_dir).joinpath(str(config_rel_path), exp_dir_suffix)
        make_dirs(exp_dir)

        # Save current config information
        with open(exp_dir.joinpath(Path(self.config_path).name), "w") as f:
            yaml.dump(OmegaConf.to_container(self.config, resolve=True), f)

        # Setup logger log path
        logger.add(
            exp_dir.joinpath("train.log"),
            format="{time} {level} {message}",
            level="INFO",
        )

        # Setup tensorboard log path
        tb_dir = exp_dir.joinpath("tbrun")
        make_dirs(tb_dir)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Setup checkpoint dir
        self.ckpt_dir = exp_dir.joinpath("checkpoint")
        make_dirs(self.ckpt_dir)

        # Setup image visualize dir
        self.log_image = exp_dir.joinpath("log_images")
        make_dirs(self.log_image)

    def to_device(self, batch: Dict[str, Union[torch.Tensor, List[str]]]):
        return {
            keyword: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for keyword, value in batch.items()
        }

    def train(self):
        if self.is_main:
            # Count parameters
            if self.use_ddp:
                num_params = count_parameters(self.pipeline.module)
            else:
                num_params = count_parameters(self.pipeline)

            for key in num_params:
                logger.info("Number of trainable parameters of {} : {}", key, num_params[key])

            if self.use_ddp:
                logger.info(f"Distributed training on {self.world_size} GPUs")

            print("Training started....")

        cur_iter = self.cur_iter or 0

        self.pipeline.train()
        total_loss = defaultdict(list)

        start_time = time.time()
        epoch = 0

        while True:
            if self.use_ddp:
                # Set epoch for sampler to ensure proper shuffling
                self.sampler.set_epoch(epoch)
                epoch += 1

            if self.continue_training and self.best_loss > 10000 and self.is_main:
                self.pipeline.eval()
                val_loss = self.run_validation(cur_iter)
                self.continue_training = False
                time_ = cal_elasped_time(time.time(), start_time)
                logger.info(
                    "Iteration: {}-Elapsed time: {}-Val loss: {}-Best loss: {}",
                    cur_iter,
                    time_,
                    val_loss,
                    self.best_loss,
                )
                self.pipeline.train()

            if self.use_ddp:
                # DDP uses standard iteration with DistributedSampler
                for batch in self.loader:
                    batch = self.to_device(batch)

                    # Perform forward and backward pass
                    loss = self.pipeline(**batch, iteration=cur_iter)

                    if loss is None:
                        continue

                    # Synchronize losses across all processes for DDP
                    for k in loss:
                        if dist.is_initialized():
                            loss_tensor = torch.tensor([loss[k].item()], device=self.device)
                            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                            total_loss[k].append(loss_tensor.item())
                        else:
                            total_loss[k].append(loss[k].item())

                    if self.config.training.update_ema:
                        self.pipeline.module.step_ema()

                    if self.is_main:
                        for k in loss:
                            self.writer.add_scalar(
                                k,
                                loss[k].item() if not dist.is_initialized() else loss_tensor.item(),
                                cur_iter + 1,
                            )

                    cur_iter += 1

                    # Logging and checkpointing (only on main process)
                    if self.is_main and (
                        cur_iter % self.config.training.log_interval == 0
                        or cur_iter % self.config.training.save_interval == 0
                    ):
                        loss_info = ""
                        for idx, k in enumerate(total_loss):
                            msg = "Train {}: {}".format(k, sum(total_loss[k]) / len(total_loss[k]))
                            if idx < len(total_loss) - 1:
                                msg += "-"
                            loss_info += msg

                    if self.is_main and cur_iter % self.config.training.log_interval == 0:
                        self.pipeline.eval()

                        if not self.config.training.skip_log_images:
                            buffer_idx = int(
                                cur_iter // self.config.training.log_interval // self.config.training.log_image_buffer
                            )
                            buffer_dir = self.log_image.joinpath(str(buffer_idx))
                            make_dirs(buffer_dir)
                            self.pipeline.module.log_images(
                                batch=self.to_device(self.batch_images),
                                save_path=buffer_dir.joinpath(f"iter_{cur_iter}.png"),
                            )

                        time_ = cal_elasped_time(time.time(), start_time)
                        logger.info(
                            "Iteration: {}-Elapsed time: {}-{}",
                            cur_iter,
                            time_,
                            loss_info,
                        )

                        self.pipeline.train()

                    if self.is_main and cur_iter % self.config.training.save_interval == 0:
                        self.pipeline.eval()

                        # Save model state dict
                        self.pipeline.module.save_state_dict(self.ckpt_dir.joinpath(f"iter_{cur_iter}.pth"))

                        if self.test_loader is not None:
                            val_loss = self.run_validation(cur_iter)
                            self.pipeline.module.save_checkpoint(
                                self.ckpt_dir.joinpath("ckpt.pth"),
                                cur_iter,
                                self.best_loss,
                            )
                            time_ = cal_elasped_time(time.time(), start_time)
                            logger.info(
                                "Iteration: {}-Elapsed time: {}-{}-Val loss: {}-Best loss: {}",
                                cur_iter,
                                time_,
                                loss_info,
                                val_loss,
                                self.best_loss,
                            )
                        else:
                            self.pipeline.module.save_checkpoint(self.ckpt_dir.joinpath("ckpt.pth"), cur_iter)

                        self.pipeline.train()

                    if cur_iter >= self.total_iterations:
                        break

                # Synchronize all processes at the end of each epoch
                if dist.is_initialized():
                    dist.barrier()
            else:
                # Original non-DDP training logic
                dataloader = iter(self.loader)

                while True:
                    try:
                        batch = next(dataloader)
                    except StopIteration:
                        dataloader = iter(self.loader)
                        batch = next(dataloader)

                    batch = self.to_device(batch)

                    # Perform forward and backward update parameter and update learning rate if has any
                    loss = self.pipeline.forward_backward_update(**batch, iteration=cur_iter)
                    if loss is None:
                        continue

                    for k in loss:
                        total_loss[k].append(loss[k].item())

                    if self.config.training.update_ema:
                        self.pipeline.step_ema()

                    for k in loss:
                        self.writer.add_scalar(k, loss[k].item(), cur_iter + 1)

                    cur_iter += 1

                    if (
                        cur_iter % self.config.training.log_interval == 0
                        or cur_iter % self.config.training.save_interval == 0
                    ):
                        loss_info = ""
                        for idx, k in enumerate(total_loss):
                            msg = "Train {}: {}".format(k, sum(total_loss[k]) / len(total_loss[k]))
                            if idx < len(total_loss) - 1:
                                msg += "-"
                            loss_info += msg

                    if cur_iter % self.config.training.log_interval == 0:
                        self.pipeline.eval()

                        if not self.config.training.skip_log_images:
                            buffer_idx = int(
                                cur_iter // self.config.training.log_interval // self.config.training.log_image_buffer
                            )
                            buffer_dir = self.log_image.joinpath(str(buffer_idx))
                            make_dirs(buffer_dir)
                            self.pipeline.log_images(
                                batch=self.to_device(self.batch_images),
                                save_path=buffer_dir.joinpath(f"iter_{cur_iter}.png"),
                            )

                        time_ = cal_elasped_time(time.time(), start_time)
                        logger.info(
                            "Iteration: {}-Elapsed time: {}-{}",
                            cur_iter,
                            time_,
                            loss_info,
                        )

                        self.pipeline.train()

                    if cur_iter % self.config.training.save_interval == 0:
                        self.pipeline.eval()

                        self.pipeline.save_state_dict(self.ckpt_dir.joinpath(f"iter_{cur_iter}.pth"))

                        if self.test_loader is not None:
                            val_loss = self.run_validation(cur_iter)
                            self.pipeline.save_checkpoint(
                                self.ckpt_dir.joinpath("ckpt.pth"),
                                cur_iter,
                                self.best_loss,
                            )
                            time_ = cal_elasped_time(time.time(), start_time)
                            logger.info(
                                "Iteration: {}-Elapsed time: {}-{}-Val loss: {}-Best loss: {}",
                                cur_iter,
                                time_,
                                loss_info,
                                val_loss,
                                self.best_loss,
                            )
                        else:
                            self.pipeline.save_checkpoint(self.ckpt_dir.joinpath("ckpt.pth"), cur_iter)

                        self.pipeline.train()

                    if cur_iter >= self.total_iterations:
                        break

            if cur_iter >= self.total_iterations:
                break

        # Synchronize all processes at the end
        if self.use_ddp and dist.is_initialized():
            dist.barrier()

    @torch.inference_mode()
    def validation_step(self, dataloader):
        val_losses = 0.0
        count = 0
        dataloader = iter(dataloader)

        while True:
            try:
                batch = next(dataloader)
            except StopIteration:
                break

            batch = self.to_device(batch)
            if self.use_ddp:
                val_loss = self.pipeline.module.validation_step(batch)
            else:
                val_loss = self.pipeline.validation_step(batch)
            val_losses += val_loss.item()
            count += 1

        final_loss = val_losses / count

        return final_loss

    @torch.inference_mode()
    def run_validation(self, iter_count):
        val_loss = self.validation_step(self.test_loader)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            if len(self.top_k_track) == self.max_top_k:
                remove_ckpt = self.top_k_track.pop(0)
                for ckpt in remove_ckpt:
                    os.remove(ckpt.as_posix())
            ckpt_path = self.ckpt_dir.joinpath(f"iter_{iter_count}_loss{self.best_loss}.pth")
            if self.config.training.update_ema:
                ema_ckpt_path = self.ckpt_dir.joinpath(f"iter_{iter_count}_loss{self.best_loss}_ema.pth")
                self.top_k_track.append((ema_ckpt_path, ckpt_path))
                if self.use_ddp:
                    self.pipeline.module.save_state_dict(ckpt_path)
                    self.pipeline.module.save_state_dict_ema(ema_ckpt_path)
                else:
                    self.pipeline.save_state_dict(ckpt_path)
                    self.pipeline.save_state_dict_ema(ema_ckpt_path)
            else:
                self.top_k_track.append((ckpt_path,))
                if self.use_ddp:
                    self.pipeline.module.save_state_dict(ckpt_path)
                else:
                    self.pipeline.save_state_dict(ckpt_path)

        return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs/constant_IAM.yaml", help="Experiment`s config")
    parser.add_argument("--ckpt_path", default=None, help="Resume training with this checkpoint")
    parser.add_argument(
        "--pretrained_path",
        default=None,
        help="Pretrained weight to initialize training",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=None,
        help="Override config values using dot-notation. Example: --config-override training.iteration=50000",
    )
    parser.add_argument("--device", default="cuda", help="Select device for training. Default: gpu")
    parser.add_argument("--ddp", action="store_true", help="Enable distributed data parallel training")
    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        args.ddp = True

    if args.ddp:
        # Setup distributed training
        rank, world_size, local_rank = setup_distributed()

        # Only set seed and create trainer on this process
        set_seed(0)

        device = f"cuda:{local_rank}"
        trainer = Trainer(
            device,
            args.config_path,
            args.ckpt_path,
            args.pretrained_path,
            args.config_override,
            rank,
            world_size,
            local_rank,
            use_ddp=True,
        )
        trainer.setup_exp()
        trainer.train()

        # Cleanup
        cleanup_distributed()
    else:
        # Single GPU training
        set_seed(0)

        device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        trainer = Trainer(
            device,
            args.config_path,
            args.ckpt_path,
            args.pretrained_path,
            args.config_override,
            use_ddp=False,
        )
        trainer.setup_exp()
        trainer.train()

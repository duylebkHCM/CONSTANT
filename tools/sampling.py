import argparse
import torch
import yaml
import time
from PIL import Image
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from helpers import (
    initialize_from_config,
    set_seed,
    make_dirs,
    make_image_grid,
    get_obj_from_str,
)
from src.data.constant import CHAR_WIDTH


class Runner:
    def __init__(self, device, weight_path, batchsize, overide_params=[]):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batchsize = batchsize

        self.root_path = Path(weight_path).parent.parent
        config_path = list(self.root_path.rglob("*.yaml"))[0]

        config = OmegaConf.load(config_path)
        self.is_overide = False
        self.test_name = Path(weight_path).stem

        if len(overide_params) > 0:
            if len(overide_params) > 0:
                overide_config = OmegaConf.from_cli(overide_params)
                config = OmegaConf.merge(config, overide_config)
                self.test_name = Path(weight_path).stem + "_".join(
                    [
                        overide_p.split("=")[-1].split("/")[-1]
                        if len(overide_p.split("=")[-1].split("/")) > 1
                        else overide_p.split("=")[-1]
                        for overide_p in overide_params
                    ]
                )
                self.is_overide = True

        self.config = config

        test_dataset = initialize_from_config(config.testdataset)
        self.random_style_image = test_dataset.random_style_image

        # use collate_fn in dataloader if your images in dataset has variable sizes
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batchsize,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            collate_fn=test_dataset.collate_function,
        )

        if not hasattr(self.config.text_encoder.params, "input_size"):
            self.config.text_encoder.params.input_size = test_dataset.tokenizer.vocab_size

        pipeline = get_obj_from_str(self.config.pipeline.target)(
            vae_config=self.config.vae.pretrained_path,
            diffusion_config=self.config.diffusion,
            style_extractor_config=self.config.style_extractor,
            text_encoder_config=self.config.text_encoder,
        )
        pipeline.load_state_dict(weight_path)
        pipeline.eval()
        self.pipeline = pipeline.to(self.device)

    def setup_logger(self):
        root_dir = Path(self.root_path).joinpath("eval", self.test_name)
        make_dirs(root_dir)
        image_dir = root_dir.joinpath("evaluate_images")
        self.real_dir = image_dir.joinpath("real")
        make_dirs(self.real_dir)
        self.gen_dir = image_dir.joinpath("gen")
        make_dirs(self.gen_dir)
        # visualize purpose only
        self.pair_dir = image_dir.joinpath("pair")
        make_dirs(self.pair_dir)
        logger.add(
            root_dir.joinpath("test.log"),
            format="{time} {level} {message}",
            level="INFO",
        )
        if self.is_overide:
            # Save current config information
            with open(root_dir.joinpath("overide.yaml"), "w") as f:
                yaml.dump(OmegaConf.to_container(self.config, resolve=True), f)

    def eval(self):
        img_idx = 0
        start_time = time.time()

        for batch in self.test_loader:
            style_images, text_embeddings, ori_wids, image_names = (
                batch["style_images"],
                batch["text_embeddings"],
                batch["ori_wids"],
                batch["image_names"],
            )
            original_lens = batch["original_lens"]

            skip = False
            for ori_wid, image_name in zip(ori_wids, image_names):
                if self.real_dir.joinpath(str(ori_wid)).joinpath(image_name).exists():
                    skip = True
                    break
            if skip:
                continue

            target_images = batch["images"].to(self.device)
            style_images = style_images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)

            input_shape = (
                self.pipeline._diffusion.img_size,
                int(text_embeddings.shape[1] * CHAR_WIDTH // self.pipeline._diffusion.down_sample),
            )

            gen_images = self.pipeline.sampling(
                input_shape,
                original_lens,
                text_embeddings,
                style_images,
                log_progress=False,
            )
            target_images = torch.unbind(target_images, dim=0)

            new_target_images = []
            for img, img_len in zip(target_images, original_lens):
                new_target_images.append(img[:, :, :img_len])
            target_images = new_target_images

            style_images = torch.unbind(style_images, dim=0)
            if not isinstance(gen_images, list):
                gen_images = list(torch.unbind(gen_images, dim=0))

            for img, ref_img, gen_img, ori_wid in zip(target_images, style_images, gen_images, ori_wids):
                img = (img + 1) * 127.5
                img = img.permute(1, 2, 0).cpu().numpy().astype("uint8")
                img = Image.fromarray(img)
                by_wid_ori = self.real_dir.joinpath(str(ori_wid))
                make_dirs(by_wid_ori)
                img_name = image_names.pop(0)

                if len(img_name.split("/")) > 1:
                    parsing_names = img_name.split("/")
                    img_name = "_".join(parsing_names[-(len(parsing_names) - 1) :])

                img.save(by_wid_ori.joinpath(img_name))

                gen_img = gen_img * 255.0
                gen_img = gen_img.permute(1, 2, 0).cpu().numpy().astype("uint8")
                gen_img = Image.fromarray(gen_img)
                by_wid_gen = self.gen_dir.joinpath(str(ori_wid))
                make_dirs(by_wid_gen)

                gen_img.save(by_wid_gen.joinpath(img_name))

                ref_img = ref_img * 255.0
                ref_img = ref_img.permute(1, 2, 0).cpu().numpy().astype("uint8")
                ref_img = Image.fromarray(ref_img)
                by_wid_pair = self.pair_dir.joinpath(str(ori_wid))
                make_dirs(by_wid_pair)

                pair_img = make_image_grid([img, gen_img, ref_img], rows=1, cols=3, resize=True, padding=5)
                pair_img.save(by_wid_pair.joinpath(str(img_idx) + ".png"))

                img_idx += 1

        elapsed_time = time.time() - start_time
        min, sec = divmod(elapsed_time, 60)
        logger.info("Complete sampling with total number of samples", img_idx)
        logger.info("Elapsed time: {} m - {} s", int(min), round(sec, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-path", required=True, help="Path to saved checkpoint")
    parser.add_argument("--batchsize", default=16, help="Batch size for sampling")
    parser.add_argument("--num_workers", default=4, help="Number of workers for dataloader")
    parser.add_argument(
        "--cond_scale",
        default=None,
        help="Overwrite guidance scale for CFG. If None, use the default value from config file binded with provided weight path",
    )
    parser.add_argument(
        "--sampling_steps",
        default=None,
        help="Overwrite samplign steps. If None, use the default value from config file binded with provided weight path",
    )
    parser.add_argument(
        "--ddim_sampling",
        default=None,
        help="Overwrite ddim sampling flag. If None, use the default value from config file binded with provided weight path",
    )
    parser.add_argument("--device", default="cuda", help="Inference device")
    args = parser.parse_args()
    set_seed(0)

    overide_params = [
        f"diffusion.params.clfree_params.cond_scale={args.cond_scale}" if args.cond_scale is not None else None,
        f"diffusion.params.sampling_steps={args.sampling_steps}" if args.sampling_steps is not None else None,
        f"diffusion.params.ddim_sampling={args.ddim_sampling}" if args.ddim_sampling is not None else None,
    ]
    overide_params = list(filter(lambda p: p is not None, overide_params))
    runner = Runner(args.device, args.weight_path, args.batchsize, overide_params)
    runner.setup_logger()
    runner.eval()

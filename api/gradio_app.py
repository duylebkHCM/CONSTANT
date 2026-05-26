from typing import Any

import gradio as gr
import torch
import random
import numpy as np
from PIL import Image
import os
from pathlib import Path
from omegaconf import OmegaConf
from helpers import initialize_from_config, get_obj_from_str
from src.data.constant import CHAR_WIDTH, FIXED_HEIGHT
import cv2


DATASET_CONFIGS = {
    "IAM": {
        "config_path": "configs/constant_IAM.yaml",
        "ckpt_path": "ckpt/IAM/ckpt.pth",
        "examples_dir": "assets/sample/IAM",
        "description": "IAM dataset",
        "preprocess_strategy": "variable",  # "variable" or "fixed"
        "fixed_size": 352,  # Only used if strategy is "fixed"
    },
    "IMGUR5K": {
        "config_path": "configs/constant_IMGUR5K.yaml",
        "ckpt_path": "ckpt/IMGUR5K/ckpt.pth",
        "examples_dir": "assets/sample/IMGUR5K",
        "description": "IMGUR5K dataset",
        "preprocess_strategy": "variable",
        "fixed_size": 352,
    },
    "IIT_English_Words": {
        "config_path": "configs/constant_IIIT.yaml",
        "ckpt_path": "ckpt/IIIT/ckpt.pth",
        "examples_dir": "assets/sample/IIIT_English",
        "description": "IIIT English Word dataset",
        "preprocess_strategy": "fixed",
        "fixed_size": 256,
    },
    "CASIA_HWDB": {
        "config_path": "configs/constant_CASIA.yaml",
        "ckpt_path": "ckpt/CASIA/ckpt.pth",
        "examples_dir": "assets/sample/CASIA_HWDB",
        "description": "CASIA_HWDB dataset",
        "preprocess_strategy": "variable",
        "fixed_size": 352,
    },
    "ViHTGen": {
        "config_path": "configs/constant_ViHTGen.yaml",
        "ckpt_path": "ckpt/ViHTGen/ckpt.pth",
        "examples_dir": "assets/sample/ViHTGen",
        "description": "ViHTGen dataset",
        "preprocess_strategy": "variable",
        "fixed_size": 352,
    },
}


class HandwritingGenerationManager:
    """Unified manager for handwriting generation pipeline."""

    def __init__(self, dataset_configs: dict):
        self.dataset_configs = dataset_configs
        self.pipelines: dict[str, Any] = {}
        self.configs: dict[str, Any] = {}
        self.tokenizers: dict[str, Any] = {}
        self.selected_reference_image = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_example_images(self, dataset_name):
        """Load example images from the dataset's examples directory."""
        examples_dir = self.dataset_configs[dataset_name]["examples_dir"]

        if not os.path.exists(examples_dir):
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = []

        for wid in sorted(Path(examples_dir).glob("*")):
            for file in list(wid.rglob("*")):
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(file.as_posix())

        # Shuffle and select 20 random images
        random.shuffle(image_files)
        return random.choices(image_files, k=20)

    def update_example_gallery(self, dataset_name):
        """Update the gallery when dataset is changed."""

        current_gallery_images = self.load_example_images(dataset_name)
        print("[DEBUG] example gallery", current_gallery_images)

        return current_gallery_images

    def on_gallery_select(self, evt: gr.SelectData, gallery_value):
        """Handle gallery image selection."""
        if evt is None:
            print("[ERROR] evt is None - SelectData not properly received")
            return None

        selected_index = evt.index

        print(f"[DEBUG] Selected index: {selected_index}")
        print(f"[DEBUG] Current gallery has {len(gallery_value)} images")

        if selected_index < len(gallery_value):
            selected_reference_image = gallery_value[selected_index]
            if isinstance(selected_reference_image, tuple):
                selected_reference_image = selected_reference_image[0]

            print(f"[DEBUG] Selected image: {selected_reference_image}")

            self.selected_reference_image = selected_reference_image

            # Return the selected image to display it
            return Image.open(self.selected_reference_image)

        return None

    def load_pipeline(self, dataset_name):
        """Load or retrieve cached pipeline for the specified dataset."""

        # Return cached pipeline if available
        if self.pipelines.get(dataset_name, None) is not None:
            print(f"[INFO] Model for {dataset_name} already loaded.")
            return self.pipelines[dataset_name], self.tokenizers[dataset_name]

        ckpt_path = self.dataset_configs[dataset_name]["ckpt_path"]

        if not os.path.exists(ckpt_path):
            raise gr.Error(
                f"Checkpoint file not found: {ckpt_path}. Please update DATASET_CONFIGS with the correct path."
            )

        print(f"[INFO] Loading {dataset_name} model from {ckpt_path}...")

        config_path = self.dataset_configs[dataset_name]["config_path"]
        config = OmegaConf.load(config_path)

        test_dataset = initialize_from_config(config.testdataset)
        tokenizer = test_dataset.tokenizer

        if not hasattr(config.text_encoder.params, "input_size"):
            config.text_encoder.params.input_size = tokenizer.vocab_size

        pipeline = get_obj_from_str(config.pipeline.target)(
            vae_config=config.vae.pretrained_path,
            diffusion_config=config.diffusion,
            style_extractor_config=config.style_extractor,
            text_encoder_config=config.text_encoder,
        )
        pipeline.load_state_dict(ckpt_path)
        pipeline.eval()
        pipeline = pipeline.to(self.device)

        # Cache the loaded components
        self.pipelines[dataset_name] = pipeline
        self.configs[dataset_name] = config
        self.tokenizers[dataset_name] = tokenizer

        return pipeline, tokenizer

    def preprocess_image(self, image_pil, dataset_name, strategy="variable", fixed_size=352):
        """
        Preprocess image with configurable strategy.

        Args:
            image_pil: PIL Image to preprocess
            dataset_name: Dataset to use configuration from
            strategy: preprocess image strategy (variable or fixed)
            fixed_size: Fixed image width if strategy = fixed
        Returns:
            Preprocessed tensor image in shape (1, C, H, W)
        """

        image_np = np.array(image_pil.convert("RGB"))
        h, w = image_np.shape[:2]

        # Resize to fixed height while maintaining aspect ratio
        new_width = int(w * (FIXED_HEIGHT / h))
        image = cv2.resize(image_np, (new_width, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)
        print(f"[DEBUG] image shape after raw resize {image.shape}")

        # Pad to multiple of CHAR_WIDTH
        pad_pixels = (CHAR_WIDTH - (new_width % CHAR_WIDTH)) % CHAR_WIDTH
        new_width += pad_pixels

        print("[DEBUG] continue to resize image so that image`s width divisible by CHAR_WIDTH")
        image = cv2.resize(image, (new_width, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)

        print(f"[DEBUG] Reference image after height resize: H={image.shape[0]}, W={new_width}")

        if strategy == "fixed":
            if new_width > fixed_size:
                # If larger, resize down to fixed_size
                image = cv2.resize(image, (fixed_size, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)
                print(f"[DEBUG] Fixed strategy: Resized from {new_width} to {fixed_size}")

            elif new_width < fixed_size:
                pad_total = fixed_size - new_width
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
                print(
                    f"[DEBUG] Fixed strategy: Padded from {new_width} to {fixed_size} (left={pad_left}, right={pad_right})"
                )

            else:
                print(f"[DEBUG] Fixed strategy: Width already matches {fixed_size}")

        elif strategy == "variable":
            # Variable strategy: Cap width at max_width (from config) but don't force resize smaller images
            max_width = self.configs[dataset_name].MAX_WIDTH
            if new_width > max_width:
                image = cv2.resize(image, (max_width, FIXED_HEIGHT), interpolation=cv2.INTER_LINEAR)
                print(f"[DEBUG] Variable strategy: Resized from {new_width} to {max_width}")

            else:
                print("[DEBUG] Variable strategy: Perform right-pad to max width")
                image = cv2.copyMakeBorder(
                    image,
                    0,
                    0,
                    0,
                    int(max_width - new_width),
                    cv2.BORDER_CONSTANT,
                    value=255,
                )

        else:
            raise ValueError(f"Unknown preprocessing strategy: {strategy}")

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image.permute(0, 3, 1, 2)

        print(f"[DEBUG] Final tensor shape: {image.shape}")

        return image

    def generate_handwriting(self, text_prompt, dataset_name, guidance_scale, sampling_steps):
        """Generate handwriting from text prompt using the selected reference image."""
        if self.selected_reference_image is None:
            raise gr.Error("Please select a reference image from the gallery.")

        if not text_prompt:
            raise gr.Error("Please enter text to generate.")

        # Load pipeline if not already loaded
        pipeline, tokenizer = self.load_pipeline(dataset_name)

        # Load image process strategy
        strategy = self.dataset_configs[dataset_name].get("preprocess_strategy", "variable")
        fixed_size = self.dataset_configs[dataset_name].get("fixed_size", 352)

        # Preprocess reference image
        ref_pil = Image.open(self.selected_reference_image).convert("RGB")
        style_image = self.preprocess_image(ref_pil, dataset_name, strategy, fixed_size).to(self.device)

        text_embedding = tokenizer.encode(text_prompt)
        text_embedding = np.array(text_embedding, dtype="int64")
        text_embedding = torch.from_numpy(text_embedding).unsqueeze(0).long().to(self.device)

        print(f"[DEBUG] Text prompt: {text_prompt}")
        print(f"[DEBUG] Text embedding shape: {text_embedding.shape}")
        print(f"[DEBUG] Text embedding: {text_embedding}")
        print(f"[DEBUG] Guidance scale: {guidance_scale}")
        print(f"[DEBUG] Sampling step: {sampling_steps}")

        # Get input and output generated shape
        if strategy == "variable":
            original_len = min(len(text_prompt) * CHAR_WIDTH, 352)
            input_shape = (
                pipeline._diffusion.img_size,
                int(text_embedding.shape[1] * CHAR_WIDTH // pipeline._diffusion.down_sample),
            )
        else:
            original_len = None
            input_shape = (
                pipeline._diffusion.img_size,
                int(fixed_size // pipeline._diffusion.down_sample),
            )

        # Generate handwriting
        with torch.inference_mode():
            generated = pipeline.sampling(
                original_lens=[original_len] if original_len else None,
                input_shape=input_shape,
                text_embeddings=text_embedding,
                style_images=style_image,
                guidance_scale=guidance_scale,
                ddim_steps=sampling_steps,
                log_progress=False,
            )

        if isinstance(generated, list) or isinstance(generated, tuple):
            generated = generated[0]

        if isinstance(generated, torch.Tensor):
            generated = generated.squeeze(0)
            generated = generated * 255.0
            generated = generated.permute(1, 2, 0).cpu().numpy().astype("uint8")

        print(f"[DEBUG] generated image shape {generated.shape}")

        return generated


manager = HandwritingGenerationManager(DATASET_CONFIGS)


def update_example_gallery(dataset_name):
    return manager.update_example_gallery(dataset_name)


def on_gallery_select(evt: gr.SelectData, gallery_value):
    return manager.on_gallery_select(evt, gallery_value)


def generate_handwriting(text_prompt, dataset_name, guidance_scale, sampling_steps):
    return manager.generate_handwriting(text_prompt, dataset_name, guidance_scale, sampling_steps)


css = """
.container { max-width: 1400px; margin: auto; padding-top: 20px; }
.header { text-align: center; margin-bottom: 30px; }
.header h1 { font-size: 2.5em; margin-bottom: 10px; }
.links { font-size: 1.1em; color: #555; }
.links a { text-decoration: none; margin: 0 10px; color: #007bff; }
#generated_image { border: 2px solid #ddd; border-radius: 8px; }
.gallery-container { height: 400px; overflow-y: auto; }
.gallery-image { cursor: pointer; transition: transform 0.2s; }
.gallery-image:hover { transform: scale(1.05); }
.selected-image-preview { border: 3px solid #007bff; border-radius: 8px; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Row(elem_classes=["header"]):
        with gr.Column():
            gr.Markdown(
                "# CONSTANT: Towards High-Quality One-Shot Handwriting Generation with Patch Contrastive Enhancement and Style-Aware Quantization"
            )
            gr.Markdown(
                """
                A state-of-the-art diffusion model for synthesizing novel text in a specific writer's style
                using only a *single* reference image.
                """
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Select Reference Image")

            dataset_dropdown = gr.Dropdown(
                choices=list(DATASET_CONFIGS.keys()),
                value=list(DATASET_CONFIGS.keys())[0],
                label="Dataset",
                info="Choose a handwriting style dataset",
            )

            example_gallery = gr.Gallery(
                value=manager.load_example_images(list(DATASET_CONFIGS.keys())[0]),
                label=f"Reference Images - {list(DATASET_CONFIGS.keys())[0]}",
                columns=4,
                height="300px",
                object_fit="contain",
                show_label=True,
                show_download_button=False,
                type="filepath",
            )

            example_reshuffle = gr.Button("Shuffle reference images", variant="primary", size="lg")

            selected_image_display = gr.Image(
                label="Selected Reference Image",
                height=100,
                show_label=True,
                elem_classes=["selected-image-preview"],
            )

            gr.Markdown("### 2. Enter Text & Generate")

            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Type the word to be generated here...",
                lines=2,
            )

            with gr.Accordion("Advanced Settings", open=False):
                guidance_slider = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0,
                    maximum=10.0,
                    value=6.0,
                    step=0.1,
                    info="Higher values align closer to reference image",
                )
                steps_slider = gr.Slider(
                    label="Sampling Steps",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                    info="More steps = higher quality, but slower",
                )

            generate_btn = gr.Button("Generate Handwriting", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Generated Output")
            output_image = gr.Image(
                label="Generated Handwriting",
                type="numpy",
                elem_id="generated_image",
                height=200,
            )

    # Update gallery when dataset changes
    dataset_dropdown.change(fn=update_example_gallery, inputs=dataset_dropdown, outputs=example_gallery)

    # Re-shuffle new list of reference images
    example_reshuffle.click(fn=update_example_gallery, inputs=dataset_dropdown, outputs=example_gallery)

    # Display selected image when gallery item is clicked
    example_gallery.select(fn=on_gallery_select, inputs=example_gallery, outputs=selected_image_display)

    # Generate button click
    generate_btn.click(
        fn=generate_handwriting,
        inputs=[text_input, dataset_dropdown, guidance_slider, steps_slider],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()

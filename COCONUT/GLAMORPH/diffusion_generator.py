# diffusion_generator.py
# Pretrained diffusion "true generation" from two outfit images
# Method: encode A,B -> VAE latents -> mix -> denoise (img2img) -> output image
#
# Requirements:
#   pip install diffusers transformers accelerate safetensors
#
# Notes:
# - Some models on HuggingFace are gated (may require login/token).
# - You can pass model_id you have access to.
# - Uses img2img with A as structural prior; mixes latents from A and B.

import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


@dataclass
class DiffusionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"  # may be gated; change if needed
    image_size: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.75
    prompt: str = "fashion garment, studio lighting, product photo, clean background"
    negative_prompt: Optional[str] = "low quality, blurry, deformed, bad anatomy, text, watermark, logo"
    seed: Optional[int] = None


_PIPE = None
_PIPE_ID = None


def _get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def _load_pipe(model_id: str):
    global _PIPE, _PIPE_ID

    if _PIPE is not None and _PIPE_ID == model_id:
        return _PIPE

    # Lazy import so the rest of the project still works without diffusers installed
    from diffusers import StableDiffusionImg2ImgPipeline

    device, dtype = _get_device_and_dtype()

    # Allow HF token via env var if model is gated
    token = os.environ.get("HF_TOKEN", None)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_auth_token=token,  # harmless if token is None; helps if gated
        safety_checker=None,   # optionally disable (research prototype)
        requires_safety_checker=False,
    ).to(device)

    # Minor speedups
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    _PIPE = pipe
    _PIPE_ID = model_id
    return pipe


def _prep_image(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size))


@torch.no_grad()
def generate_fused_image(
    image_path_a: str,
    image_path_b: str,
    out_path: str,
    alpha: float,
    cfg: Optional[DiffusionConfig] = None,
) -> str:
    """
    Generates a NEW garment image using pretrained diffusion.

    Args:
        image_path_a: parent A local path
        image_path_b: parent B local path
        out_path: where to save the generated image
        alpha: blend weight (towards A). 0.0->B, 1.0->A
        cfg: DiffusionConfig

    Returns:
        out_path
    """
    if cfg is None:
        cfg = DiffusionConfig()

    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pipe = _load_pipe(cfg.model_id)
    device, dtype = _get_device_and_dtype()

    imgA = _prep_image(image_path_a, cfg.image_size)
    imgB = _prep_image(image_path_b, cfg.image_size)

    # Set seed for reproducibility
    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

    # Encode to VAE latents (mean)
    a = pipe.image_processor.preprocess(imgA).to(device=device, dtype=dtype)
    b = pipe.image_processor.preprocess(imgB).to(device=device, dtype=dtype)

    latA = pipe.vae.encode(a).latent_dist.mean
    latB = pipe.vae.encode(b).latent_dist.mean

    # Latent fusion
    lat_mix = alpha * latA + (1.0 - alpha) * latB

    # Denoise starting from the mixed latent, using A as structure prior
    result = pipe(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        image=imgA,
        strength=cfg.strength,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.steps,
        latents=lat_mix,
        generator=generator,
    ).images[0]

    result.save(out_path)
    return out_path

"""Photoreal image generation with SDXL (RealVisXL V5.0). Runs on CUDA when there is a GPU, else on the CPU
(GitHub-hosted runners have no GPU: float32 there, about 11 GB of RAM).

  python gen.py txt2img --prompt "..." --out DIR --name NAME [--seeds 1 2] [--w 1024 --h 768]
  python gen.py inpaint --image after.png --mask mask.png --prompt "..." --out DIR --name NAME [--strength 0.95]

Mask: white = repaint, black = keep. After inpainting, the black area is pasted back from the source
pixel-for-pixel (feathered seam), so a before/after pair lines up exactly.
"""
import argparse
import gc
import os
import pathlib
import time

import torch
from PIL import Image, ImageFilter
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline

MODEL = os.environ.get("MODEL_PATH", "models/RealVisXL_V5.0_fp16.safetensors")
NEGATIVE = (
    "cgi, 3d render, cartoon, illustration, painting, drawing, anime, plastic, oversaturated, "
    "text, watermark, logo, signature, people, person, hands, blurry, lowres, jpeg artifacts, "
    "deformed, distorted, warped lines, duplicate, frame, border"
)
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"


def load(mode: str):
    """Load only the pipeline this mode needs. (Building the inpaint pipeline from a txt2img one with
    from_pipe kept a second set of references and pushed a 16 GB runner into swap: 90 s per step.)

    CPU: the fp16 file is loaded as it is (7 GB) and each part is upcast to float32 only when it is used,
    see embed(). bfloat16 is not an option there: without native bf16 instructions PyTorch runs it in a
    slow fallback path (see final/NOTES.md)."""
    if not CUDA:
        torch.set_num_threads(os.cpu_count() or 4)
    cls = StableDiffusionXLInpaintPipeline if mode == "inpaint" else StableDiffusionXLPipeline
    pipe = cls.from_single_file(MODEL, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.vae.config.force_upcast = True  # decode in fp32: no black images
    if CUDA:
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
    return pipe


def embed(pipe, prompt: str, negative: str) -> dict:
    """Encode the prompt once. On the CPU the text encoders are then dropped and the UNet and VAE are upcast
    to float32, so the float32 run peaks near 11 GB instead of 14 GB (the runner has 16 GB)."""
    if not CUDA:
        pipe.text_encoder.to(torch.float32)
        pipe.text_encoder_2.to(torch.float32)
    with torch.no_grad():
        pe, ne, ppe, npe = pipe.encode_prompt(prompt, negative_prompt=negative, device=DEVICE,
                                              do_classifier_free_guidance=True)
    if not CUDA:
        pipe.text_encoder = pipe.text_encoder_2 = None
        gc.collect()
        pipe.unet.to(torch.float32)
        pipe.vae.to(torch.float32)
    return dict(prompt_embeds=pe, negative_prompt_embeds=ne, pooled_prompt_embeds=ppe,
                negative_pooled_prompt_embeds=npe)


def run(a):
    pipe = load(a.mode)
    emb = embed(pipe, a.prompt, a.negative or NEGATIVE)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = mask = soft = None
    if a.mode == "inpaint":
        src = Image.open(a.image).convert("RGB")
        mask = Image.open(a.mask).convert("L").resize(src.size)
        soft = mask.filter(ImageFilter.GaussianBlur(a.feather))
    for seed in a.seeds:
        t = time.time()
        g = torch.Generator(DEVICE).manual_seed(seed)
        common = dict(num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g, **emb)
        if a.mode == "txt2img":
            img = pipe(width=a.w, height=a.h, **common).images[0]
        else:
            img = pipe(image=src, mask_image=mask, width=src.width, height=src.height,
                       strength=a.strength, **common).images[0].resize(src.size)
            img = Image.composite(img, src, soft)
        p = out / f"{a.name}_s{seed}.png"
        img.save(p)
        print(f"saved {p} ({time.time() - t:.0f}s on {DEVICE})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["txt2img", "inpaint"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="img")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=768)
    ap.add_argument("--image")
    ap.add_argument("--mask")
    ap.add_argument("--strength", type=float, default=0.95)
    ap.add_argument("--feather", type=float, default=6)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

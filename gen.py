"""Photoreal image generation with SDXL (RealVisXL V5.0). Runs on CUDA when there is a GPU, else on the CPU
(GitHub-hosted runners have no GPU: bfloat16 on the CPU keeps RAM around 8 GB).

  python gen.py txt2img --prompt "..." --out DIR --name NAME [--seeds 1 2] [--w 1024 --h 768]
  python gen.py inpaint --image after.png --mask mask.png --prompt "..." --out DIR --name NAME [--strength 0.95]

Mask: white = repaint, black = keep. After inpainting, the black area is pasted back from the source
pixel-for-pixel (feathered seam), so a before/after pair lines up exactly.
"""
import argparse
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
DTYPE = torch.float16 if CUDA else torch.bfloat16


def load_base() -> StableDiffusionXLPipeline:
    if not CUDA:
        torch.set_num_threads(os.cpu_count() or 4)
    pipe = StableDiffusionXLPipeline.from_single_file(MODEL, torch_dtype=DTYPE, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.vae.config.force_upcast = True  # decode in fp32: no black images
    if CUDA:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
    return pipe


def run(a):
    base = load_base()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = mask = soft = None
    pipe = base
    if a.mode == "inpaint":
        pipe = StableDiffusionXLInpaintPipeline.from_pipe(base)
        if CUDA:
            pipe.enable_model_cpu_offload()
        src = Image.open(a.image).convert("RGB")
        mask = Image.open(a.mask).convert("L").resize(src.size)
        soft = mask.filter(ImageFilter.GaussianBlur(a.feather))
    for seed in a.seeds:
        t = time.time()
        g = torch.Generator(DEVICE).manual_seed(seed)
        common = dict(prompt=a.prompt, negative_prompt=a.negative or NEGATIVE,
                      num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g)
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

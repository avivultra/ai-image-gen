"""Speed diagnostic for the CPU runner: is bfloat16 slow on this CPU, and how fast is inpaint in each dtype?

  python diag.py micro            # matmul / conv timings, float32 vs bfloat16 (no model needed)
  python diag.py model float32    # real inpaint pipeline: VAE encode + a few steps at 512x384 (and 1024x768)
"""
import gc
import os
import subprocess
import sys
import time

import torch

torch.set_num_threads(os.cpu_count() or 4)


def cpu_info():
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(("Model name", "CPU(s)", "Thread", "Core")):
            print(line)
        if line.startswith("Flags"):
            flags = line.split(":", 1)[1].split()
            print("bf16/amx flags:", [f for f in flags if "bf16" in f or "amx" in f or f == "avx512f"])
    print("torch", torch.__version__, "threads", torch.get_num_threads())
    print("mkldnn bf16 supported:", torch.ops.mkldnn._is_mkldnn_bf16_supported())


def timeit(fn, n=3):
    fn()
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n


def micro():
    for dt in (torch.float32, torch.bfloat16):
        a, b = torch.randn(2048, 2048, dtype=dt), torch.randn(2048, 2048, dtype=dt)
        conv = torch.nn.Conv2d(320, 320, 3, padding=1).to(dt)
        x = torch.randn(2, 320, 48, 64, dtype=dt)
        with torch.no_grad():
            print(f"{dt}: matmul 2048^2 {timeit(lambda: a @ b) * 1000:.0f} ms, "
                  f"conv 320x48x64 {timeit(lambda: conv(x)) * 1000:.0f} ms", flush=True)


def model(dtype_name):
    from PIL import Image
    from diffusers import StableDiffusionXLInpaintPipeline

    dt = getattr(torch, dtype_name)
    t = time.time()
    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        "models/RealVisXL_V5.0_fp16.safetensors", torch_dtype=dt, use_safetensors=True)
    pipe.vae.config.force_upcast = True
    pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
    print(f"[{dtype_name}] loaded in {time.time() - t:.0f}s", flush=True)

    src = Image.open("outputs/sink_after/sink_after_s22.png").convert("RGB")
    mask = Image.open("masks/sink.png").convert("L")
    sizes = [(512, 384), (1024, 768)] if dt == torch.float32 else [(512, 384)]
    for w, h in sizes:
        img = src.resize((w, h))
        with torch.no_grad():
            x = pipe.image_processor.preprocess(img).to(torch.float32)
            vae32 = pipe.vae.to(torch.float32)
            t = time.time()
            vae32.encode(x).latent_dist.mean
            print(f"[{dtype_name}] {w}x{h} VAE encode (fp32): {time.time() - t:.1f}s", flush=True)
            pipe.vae.to(dt)
        stamps = [time.time()]

        def cb(p, i, ts, kw):
            stamps.append(time.time())
            print(f"[{dtype_name}] {w}x{h} step {i + 1}: {stamps[-1] - stamps[-2]:.1f}s", flush=True)
            return kw

        t = time.time()
        pipe(prompt="photo of an old bathroom faucet", image=img, mask_image=mask.resize((w, h)),
             width=w, height=h, strength=0.95, num_inference_steps=4, guidance_scale=4.5,
             callback_on_step_end=cb, generator=torch.Generator("cpu").manual_seed(1))
        print(f"[{dtype_name}] {w}x{h} whole call (4 steps): {time.time() - t:.0f}s", flush=True)
    del pipe
    gc.collect()


if __name__ == "__main__":
    cpu_info()
    if sys.argv[1] == "micro":
        micro()
    else:
        model(sys.argv[2])

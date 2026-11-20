"""Speed diagnostic for the CPU runner: which stage of an inpaint run is slow, and in which dtype?

  python diag.py micro            # tiny matmul / conv timings, float32 vs bfloat16 (no model needed)
  python diag.py model float32    # text encode, VAE encode, UNet forward (512x384 and 1024x768), RAM
  python diag.py model bfloat16   # the same stages at 512x384 only; run it under a timeout

Every line is printed as soon as it is measured, so a stage that hangs is the one after the last line.
"""
import os
import subprocess
import sys
import time

import torch

MODEL = "models/RealVisXL_V5.0_fp16.safetensors"
torch.set_num_threads(os.cpu_count() or 4)


def rss():
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            yield line.split(":")[0] + " " + str(int(line.split()[1]) // 1024) + " MB"


def say(msg):
    print(f"{msg}   [{', '.join(rss())}]", flush=True)


def cpu_info():
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(("Model name", "CPU(s)", "Thread", "Core")):
            print(line)
        if line.startswith("Flags"):
            flags = line.split(":", 1)[1].split()
            print("bf16/amx/avx512 flags:", [f for f in flags if "bf16" in f or "amx" in f or f == "avx512f"])
    print("torch", torch.__version__, "threads", torch.get_num_threads())
    print("mkldnn bf16 supported:", torch.ops.mkldnn._is_mkldnn_bf16_supported(), flush=True)


def timeit(fn, n=2):
    fn()
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n


def micro():
    for threads in (2, os.cpu_count() or 4):
        torch.set_num_threads(threads)
        for dt in (torch.float32, torch.bfloat16):
            a, b = torch.randn(512, 512, dtype=dt), torch.randn(512, 512, dtype=dt)
            conv = torch.nn.Conv2d(64, 64, 3, padding=1).to(dt)
            x = torch.randn(1, 64, 32, 32, dtype=dt)
            with torch.no_grad():
                print(f"{threads} threads, {dt}: matmul 512^2 {timeit(lambda: a @ b) * 1000:.1f} ms", flush=True)
                print(f"{threads} threads, {dt}: conv 64ch 32x32 {timeit(lambda: conv(x)) * 1000:.1f} ms", flush=True)


def model(dtype_name):
    from PIL import Image
    from diffusers import StableDiffusionXLInpaintPipeline

    dt = getattr(torch, dtype_name)
    fp32 = dt == torch.float32
    t = time.time()
    # float32: load the fp16 file as it is (7 GB), then upcast one component at a time
    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        MODEL, torch_dtype=torch.float16 if fp32 else dt, use_safetensors=True)
    say(f"[{dtype_name}] loaded in {time.time() - t:.0f}s")

    with torch.no_grad():
        t = time.time()
        if fp32:
            pipe.text_encoder.to(torch.float32)
            pipe.text_encoder_2.to(torch.float32)
        emb = pipe.encode_prompt("photo of an old bathroom faucet", negative_prompt="cartoon",
                                 device="cpu", do_classifier_free_guidance=True)
        say(f"[{dtype_name}] text encode (both encoders, cond + uncond): {time.time() - t:.1f}s")
        pipe.text_encoder = pipe.text_encoder_2 = None
        import gc
        gc.collect()
        say(f"[{dtype_name}] text encoders dropped")

        src = Image.open("outputs/sink_after/sink_after_s22.png").convert("RGB")
        pipe.vae.to(torch.float32)
        for w, h in ([(512, 384), (1024, 768)] if fp32 else [(512, 384)]):
            x = pipe.image_processor.preprocess(src.resize((w, h))).to(torch.float32)
            t = time.time()
            pipe.vae.encode(x).latent_dist.mean
            say(f"[{dtype_name}] {w}x{h} VAE encode in fp32: {time.time() - t:.1f}s")

        t = time.time()
        pipe.unet.to(dt)
        say(f"[{dtype_name}] unet.to({dtype_name}): {time.time() - t:.0f}s")
        pe = torch.cat([emb[1], emb[0]]).to(dt)
        te = torch.cat([emb[3], emb[2]]).to(dt)
        ids = torch.tensor([[768.0, 1024, 0, 0, 768, 1024]] * 2, dtype=dt)

        def step(w, h):
            lat = torch.randn(2, 4, h // 8, w // 8, dtype=dt)
            t = time.time()
            pipe.unet(lat, 500, encoder_hidden_states=pe, added_cond_kwargs={"text_embeds": te, "time_ids": ids})
            return time.time() - t

        for w, h in ([(512, 384), (1024, 768)] if fp32 else [(512, 384)]):
            say(f"[{dtype_name}] {w}x{h} UNet forward, batch 2 (= one CFG step): {step(w, h):.1f}s")
            say(f"[{dtype_name}] {w}x{h} UNet forward again: {step(w, h):.1f}s")
        pipe.enable_attention_slicing()
        say(f"[{dtype_name}] 512x384 UNet forward with attention slicing: {step(512, 384):.1f}s")
        if fp32:
            say(f"[{dtype_name}] 1024x768 UNet forward with attention slicing: {step(1024, 768):.1f}s")


if __name__ == "__main__":
    cpu_info()
    if sys.argv[1] == "micro":
        micro()
    else:
        model(sys.argv[2])

"""Run every job in jobs/*.json whose images are not all in outputs/<job name>/ yet. Called by the GitHub workflow.

A job file:
  {"mode": "txt2img", "prompt": "...", "seeds": [1, 2], "w": 1024, "h": 768, "steps": 28, "cfg": 4.5}
  {"mode": "inpaint", "image": "outputs/sink_after/sink_after_s1.png", "mask": "masks/sink.png",
   "prompt": "...", "seeds": [1, 2], "strength": 0.95, "feather": 6}
Results land in outputs/<job name>/<job name>_s<seed>.png. One seed runs at a time, and a seed whose PNG
already exists is skipped, so a cancelled run resumes where it stopped. With COMMIT_EACH=1 (the workflow)
every finished image is committed and pushed at once, so nothing is lost if the run is cut short.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def push(path: pathlib.Path, msg: str):
    subprocess.run(["git", "add", str(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
    subprocess.run(["git", "pull", "-q", "--rebase", "--autostash"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)


ran = 0
for job in sorted((ROOT / "jobs").glob("*.json")):
    name = job.stem
    out = ROOT / "outputs" / name
    j = json.loads(job.read_text(encoding="utf-8"))
    for seed in j.get("seeds", [1]):
        png = out / f"{name}_s{seed}.png"
        if png.exists():
            continue
        cmd = [sys.executable, str(ROOT / "gen.py"), j["mode"], "--prompt", j["prompt"], "--out", str(out),
               "--name", name, "--seeds", str(seed)]
        for k in ("negative", "steps", "cfg", "w", "h", "strength", "feather"):
            if k in j:
                cmd += [f"--{k}", str(j[k])]
        if j["mode"] == "inpaint":
            cmd += ["--image", str(ROOT / j["image"]), "--mask", str(ROOT / j["mask"])]
        print(f"== {name} seed {seed}", flush=True)
        subprocess.run(cmd, check=True)
        ran += 1
        if os.environ.get("COMMIT_EACH") == "1":
            push(png, f"Generated {png.name} [skip ci]")
print(f"done: {ran} image(s) generated")

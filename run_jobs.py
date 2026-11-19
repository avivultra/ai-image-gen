"""Run every job in jobs/*.json that has no outputs/<job name>/ folder yet. Called by the GitHub workflow.

A job file:
  {"mode": "txt2img", "prompt": "...", "seeds": [1, 2], "w": 1024, "h": 768, "steps": 28, "cfg": 4.5}
  {"mode": "inpaint", "image": "outputs/sink_after/sink_after_s1.png", "mask": "masks/sink.png",
   "prompt": "...", "seeds": [1, 2], "strength": 0.95}
Results land in outputs/<job name>/<job name>_s<seed>.png. To re-run a job, delete its outputs folder
(or copy the job under a new name with other seeds / prompt).
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
ran = 0
for job in sorted((ROOT / "jobs").glob("*.json")):
    name = job.stem
    out = ROOT / "outputs" / name
    if out.exists():
        continue
    j = json.loads(job.read_text(encoding="utf-8"))
    cmd = [sys.executable, str(ROOT / "gen.py"), j["mode"], "--prompt", j["prompt"], "--out", str(out), "--name", name,
           "--seeds", *[str(s) for s in j.get("seeds", [1])]]
    for k in ("negative", "steps", "cfg", "w", "h", "strength", "feather"):
        if k in j:
            cmd += [f"--{k}", str(j[k])]
    if j["mode"] == "inpaint":
        cmd += ["--image", str(ROOT / j["image"]), "--mask", str(ROOT / j["mask"])]
    print(f"== {name}", flush=True)
    subprocess.run(cmd, check=True)
    ran += 1
print(f"done: {ran} job(s) ran")

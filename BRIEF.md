# Brief: before / after photos for a handyman's website

You are continuing this task in a cloud session. Everything you need is in this repository.
Nothing here belongs to the website itself; only the four final photos leave this repo.

## Status (read this first)

**Done.** The four finals are in `final/`, with the sources, seeds, known flaws and the fix for the slow
CPU runs in `final/NOTES.md`. The rest of this file is the original brief.

## The goal

Two before/after pairs that look like **real phone or camera photos**, not renders.
The site shows them in a drag slider, so each pair must be the **same scene, same angle,
pixel-aligned**; only the repaired part changes.

| Pair | After (the finished job) | Before (the same scene, broken) |
|---|---|---|
| `sink` | Clean vanity: new brushed-chrome faucet, crisp white silicone line at the backsplash | Old corroded faucet with limescale and a drip, black moldy cracked silicone peeling at the joint, hard-water stains around the faucet base |
| `wall` | Smooth freshly painted wall, an oak floating shelf with a few books and a small vase | The same wall with no shelf: a long hairline crack in the plaster, a few old screw holes and patched spots, scuffs and grey stains, slightly dingy paint |

Quality bar: believable light and materials, straight tile lines, no warped geometry,
no text, no people or hands, nothing that screams AI. Ordinary, slightly imperfect homes
read as more real than showroom perfection.

## How generation works here

- `jobs/<name>.json` describes a job. **Pushing a new job file starts the `generate` workflow**
  (`.github/workflows/generate.yml`). It runs `run_jobs.py`, which generates every job whose
  `outputs/<name>/` folder does not exist yet, then commits the PNGs to `outputs/<name>/`.
- GitHub runners have **no GPU**, so this is CPU work: about 10-30 minutes per image.
  Put 2 seeds per job, not more. Check progress with `git pull` (or the Actions tab).
- Job formats are documented at the top of `run_jobs.py`. To retry, add a new job file
  with a new name (e.g. `sink_after_v2.json`); do not delete outputs you might still pick.
- `candidates/sink_after_local_s11.png` was already made on a GPU with the same model and
  the `sink_after` prompt. It is a valid candidate for the sink "after".

## Steps

1. (Done) Afters chosen, see Status.
2. (Done) For each chosen "after", make a mask with Pillow, `masks/<pair>.png`, the same size as the image:
   white where the "before" must differ, black everywhere else.
   - sink: the faucet and its base, the silicone joint along the backsplash, a band of countertop around them.
   - wall: the shelf and its contents (they must disappear) plus the wall areas where the crack, holes and stains go.
     Keep the floor and baseboard black.
3. Queue the inpaint jobs, e.g. `jobs/sink_before.json`:
   ```json
   {"mode": "inpaint", "image": "outputs/sink_after/sink_after_s22.png", "mask": "masks/sink.png",
    "prompt": "<the broken state, photo style words as in the after prompt>", "seeds": [5, 6], "strength": 0.95}
   ```
   `gen.py` pastes the unmasked area back pixel-for-pixel, so alignment is guaranteed.
4. Pick the best "before" for each pair. Check the pair side by side: same light, no visible seam.
5. Export the finals to `final/`: `sink-after.jpg`, `sink-before.jpg`, `wall-after.jpg`, `wall-before.jpg`,
   exactly **1200x900**, JPEG quality 85, sRGB (resize with LANCZOS; the source is 1024x768, same 4:3).
6. Commit `final/` with a short message and write a short `final/NOTES.md`: which source files and
   seeds were used, and anything that is not perfect.

## Rules

- Do not add anything about the website, its owner or customers to this repo (it is public).
- Do not change `gen.py` sampling defaults unless a result clearly needs it; say so in `NOTES.md` if you do.
- No other models or paid services.

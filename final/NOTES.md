# Final photos

All four are 1200x900 JPEG, quality 85, sRGB, resized with LANCZOS from 1024x768 sources.

| File | Source | Seed |
|---|---|---|
| `sink-after.jpg` | `outputs/sink_after/sink_after_s22.png` (txt2img) | 22 |
| `sink-before.jpg` | `outputs/sink_before_t1/sink_before_t1_s5.png` | 5 (on top of `sink_before_v2_s5`, seed 5) |
| `wall-after.jpg` | `outputs/wall_after/wall_after_s1.png` (txt2img) | 1 |
| `wall-before.jpg` | `outputs/wall_before_t1/wall_before_t1_s6.png` | 6 (on top of `wall_before_v2_s5`, seed 5) |

## How the befores were made

Each before took two inpaint passes over the after, both with the unchanged `gen.py` sampling defaults
(28 steps, cfg 4.5); only strength, feather, prompt and negative prompt are set per job in `jobs/`.

1. `*_before_v2` (strength 0.55): the damage is first sketched into the image with Pillow
   (`make_masks.py` -> `inputs/sink_sketch.png`, `inputs/wall_sketch.png`), and the model turns the sketch
   into a photo. For the wall, the shelf, the frame and the shelf shadow are removed procedurally before that.
2. `*_before_t1` (strength 0.4 sink / 0.35 wall, small masks): the first pass had cleaned away the screw holes
   and most of the mold, so those were sketched again and repainted locally.

Rejected: `sink_before`, `wall_before` (first round, strength 0.95 / 0.85 without a sketch: moss-green faucet and
green silicone; the repainted wall box came out in a different tone with a hard edge), `*_before_v3`
(strength 0.7: the faucet changes shape, the crack dissolves), `sink_before_t2`, the other seeds.

## Checks on the pairs

- Alignment: before and after are the same pixels outside the repainted areas (compared numerically, difference
  of at most 2 levels): sink 85% of the frame identical, wall 55% (the floor, baseboard, window, radiator and
  the wall right of the shelf).
- Light: same direction and colour; the wall before is brighter where the shelf and its shadow used to be,
  as it should be.

## Not perfect

- Sink: the mold sits in fairly regular dots along the joint; at full size it can read as slightly "placed".
  The before faucet is the same model as the after, only dull and limescaled, with a slightly different lever.
  There is no visible drip.
- Wall: a very faint vertical tone step in the bare wall about three quarters of the way across (the right
  edge of the removed shelf area), visible only on a slow drag of the slider. The filler patches from the brief
  are barely there; the crack, the screw holes and the faint stains carry the picture.
- The wall crack is clear but a little heavier than "hairline".

## The speed problem and the change in gen.py

Cause: GitHub hands out two kinds of CPU runner, and `gen.py` always used bfloat16 on the CPU.

| Runner CPU | native bf16 | UNet step 512x384, float32 | UNet step 512x384, bfloat16 |
|---|---|---|---|
| AMD EPYC 9V45 | yes (`avx512_bf16`) | 12.6 s | 3.3 s |
| AMD EPYC 7763 | no | about 19 s | about 490 s |

On the EPYC 7763 PyTorch has no fast bfloat16 path (a 2048x2048 matmul: 106 ms in float32, 59 s in bfloat16).
The fast txt2img run had landed on a bf16 machine, the slow inpaint runs on the other kind; inpainting itself
was not the problem. The VAE encode is 10-26 s at 1024x768 on either. Measured with `diag.py` / the `diag` workflow.

Change in `gen.py` (no sampling default touched):
- On the CPU it now asks PyTorch whether bf16 is native. If yes: bfloat16 as before. If not: float32, loading
  the fp16 file as it is and upcasting one component at a time.
- The prompt is encoded once and the text encoders are dropped before the UNet runs, so the float32 run
  stays near 11 GB on the 16 GB runner.

Resulting time per 1024x768 inpaint image: about 12 minutes on a bf16 runner, about 36 minutes on the other.

Workflow: `generate` takes an optional `job` input (run one job file only), so several jobs can run side by side.

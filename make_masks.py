"""Build the inpaint masks (white = the "before" differs, black = keep) and the wall pre-fill image.

  python make_masks.py            -> masks/sink.png, masks/wall.png, inputs/wall_prefill.png
  python make_masks.py --preview  -> also writes preview_*.png overlays (not committed)

Wall: the shelf, its contents and the picture frame must vanish. A plain SDXL checkpoint struggles to
erase a large object, so the masked area is first pre-filled with the surrounding wall (column-wise
interpolation between the bare wall above and below the shelf: the sides hold shadows). The inpaint then starts from a bare
wall and only has to add the crack, holes and stains, which keeps the paint tone continuous.
"""
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parent
SINK = ROOT / "outputs/sink_after/sink_after_s22.png"
WALL = ROOT / "outputs/wall_after/wall_after_s1.png"
WALL_BOX = (0, 12, 765, 458)  # shelf, its contents, the frame and the shelf shadow (it reaches the left edge)


def sink_mask(size):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rectangle((490, 170, 800, 500), fill=255)  # faucet
    d.ellipse((610, 450, 860, 525), fill=255)  # countertop around the faucet base
    # silicone joint along the backsplash, with a gap at the vase so its glass base is not repainted
    d.line([(0, 405), (270, 411)], fill=255, width=40)
    d.line([(395, 420), (520, 440), (650, 458), (800, 470), (1024, 500)], fill=255, width=46, joint="curve")
    return m


def wall_mask(size):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(WALL_BOX, fill=255)
    return m


def wall_prefill(img, box):
    a = np.asarray(img).astype(np.float32)
    x0, y0, x1, y1 = box
    top = a[max(0, y0 - 10):y0 - 2, x0:x1].mean(axis=0)  # bare wall just above the box, per column
    bot = a[y1 + 4:y1 + 14, x0:x1].mean(axis=0)  # bare wall just below
    k = np.ones(31) / 31  # smooth along x so single pixels do not streak
    top = np.stack([np.convolve(np.pad(top[:, c], 15, mode="edge"), k, "valid") for c in range(3)], 1)
    bot = np.stack([np.convolve(np.pad(bot[:, c], 15, mode="edge"), k, "valid") for c in range(3)], 1)
    t = np.linspace(0, 1, y1 - y0)[:, None, None]
    fill = top[None] * (1 - t) + bot[None] * t
    fill += np.random.default_rng(0).normal(0, 1.2, fill.shape)  # a little grain, like the photo
    a[y0:y1, x0:x1] = fill
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def main():
    (ROOT / "masks").mkdir(exist_ok=True)
    (ROOT / "inputs").mkdir(exist_ok=True)
    sink = Image.open(SINK).convert("RGB")
    wall = Image.open(WALL).convert("RGB")
    sm, wm = sink_mask(sink.size), wall_mask(wall.size)
    sm.save(ROOT / "masks/sink.png")
    wm.save(ROOT / "masks/wall.png")
    pre = wall_prefill(wall, WALL_BOX)
    pre = Image.composite(pre, wall, wm.filter(ImageFilter.GaussianBlur(4)))
    pre.save(ROOT / "inputs/wall_prefill.png")
    if "--preview" in sys.argv:
        red = Image.new("RGB", sink.size, (255, 0, 0))
        Image.blend(sink, Image.composite(red, sink, sm), 0.45).save(ROOT / "preview_sink.png")
        Image.blend(wall, Image.composite(red, wall, wm), 0.45).save(ROOT / "preview_wall.png")
    print("masks written")


if __name__ == "__main__":
    main()

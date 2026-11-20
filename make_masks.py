"""Build the inpaint masks (white = the "before" differs, black = keep) and the wall pre-fill image.

  python make_masks.py            -> masks/sink.png, masks/wall.png, inputs/wall_prefill.png
  python make_masks.py --preview  -> also writes preview_*.png overlays (not committed)

Second round (v2), after the first befores came out overdone (a moss-green faucet, a wall box in another tone):
the damage is first sketched roughly into the image with Pillow (inputs/sink_sketch.png, inputs/wall_sketch.png)
and the inpaint runs at a lower strength over a tighter mask (masks/sink_v2.png, masks/wall_v2.png), so the
model only has to turn the sketch into a photo and cannot repaint the scene. The wall fill for v2 also reaches
below the shelf shadow, which the first pre-fill had sampled as "bare wall".

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


def wall_clean(img):
    """The wall with the shelf, the frame and the shelf shadow removed: every column is interpolated between
    the bare wall at the top edge and the bare wall below the shadow (which ends near y=545 on the left and
    is absent right of x=700)."""
    a = np.asarray(img).astype(np.float32)
    h, w, _ = a.shape
    x0, y0, x1 = 0, 12, 772
    yb = np.interp(np.arange(x0, x1), [0, 600, 700, w], [548, 548, 464, 464]).astype(int)
    k = np.ones(41) / 41

    def smooth(v):
        return np.stack([np.convolve(np.pad(v[:, c], 20, mode="edge"), k, "valid") for c in range(3)], 1)

    top = smooth(a[2:10, x0:x1].mean(axis=0))
    bot = smooth(np.stack([a[yb[i] + 4:yb[i] + 16, x0 + i].mean(axis=0) for i in range(x1 - x0)]))
    rng = np.random.default_rng(0)
    out = a.copy()
    m = np.zeros((h, w), np.float32)
    for i in range(x1 - x0):
        n = yb[i] - y0
        t = np.linspace(0, 1, n)[:, None]
        out[y0:yb[i], x0 + i] = top[i] * (1 - t) + bot[i] * t
        m[y0:yb[i], x0 + i] = 1
    grain = np.asarray(Image.fromarray((rng.normal(0, 1, (h, w)) * 40 + 128).clip(0, 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32) - 128
    out += grain[..., None] * 0.09 * m[..., None]
    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))) / 255.0
    out = out * soft[..., None] + a * (1 - soft[..., None])
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def blotches(size, rng, pts, r, n, spread):
    """A soft-edged alpha layer of n small irregular spots scattered around the given points."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for _ in range(n):
        x, y = pts[rng.integers(len(pts))]
        x += rng.normal(0, spread[0])
        y += rng.normal(0, spread[1])
        rr = r * rng.uniform(0.4, 1.3)
        d.ellipse((x - rr, y - rr * rng.uniform(0.5, 1), x + rr, y + rr * rng.uniform(0.5, 1)),
                  fill=int(rng.uniform(120, 255)))
    return m


def along(points, step=3):
    out = []
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        n = max(2, int(np.hypot(xb - xa, yb - ya) / step))
        out += [(xa + (xb - xa) * i / n, ya + (yb - ya) * i / n) for i in range(n)]
    return out


def wall_v2(wall):
    """Sketch a hairline crack, old screw holes, filler patches and scuffs on the cleaned wall."""
    rng = np.random.default_rng(3)
    img = wall_clean(wall)
    size = img.size
    mask = Image.new("L", size, 0)
    md = ImageDraw.Draw(mask)
    # scuffs and grey stains: faint, wide
    stain_pts = [(150, 330), (380, 250), (560, 150), (90, 120), (640, 420), (300, 470)]
    st = blotches(size, rng, stain_pts, 26, 26, (30, 18)).filter(ImageFilter.GaussianBlur(9))
    img = Image.composite(Image.new("RGB", size, (120, 114, 108)), img, st.point(lambda v: int(v * 0.16)))
    for x, y in stain_pts:
        md.ellipse((x - 85, y - 55, x + 85, y + 55), fill=255)
    # filler patches: slightly lighter, matte, where the shelf brackets and the frame hung
    holes = [(262, 392), (455, 396), (668, 390), (478, 38)]
    patch = Image.new("L", size, 0)
    pd = ImageDraw.Draw(patch)
    for x, y in holes[:2]:
        pd.ellipse((x - 20, y - 13, x + 16, y + 15), fill=255)
    patch = patch.filter(ImageFilter.GaussianBlur(3))
    img = Image.composite(Image.new("RGB", size, (226, 222, 214)), img, patch.point(lambda v: int(v * 0.5)))
    # screw holes: a dark dot with a small chipped rim
    d = ImageDraw.Draw(img)
    for x, y in holes + [(300, 392), (640, 392)]:
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(205, 200, 192))
        d.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=(58, 52, 48))
        md.ellipse((x - 40, y - 40, x + 40, y + 40), fill=255)
    # hairline crack: a jittered diagonal with two short branches
    def crack(a, b, n, jit):
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n) + np.cumsum(rng.normal(0, jit, n)) * 0.5
        xs = xs + rng.normal(0, jit * 0.6, n)
        return list(zip(xs, ys))

    main = crack((70, 78), (610, 352), 60, 3.0)
    for line, wd in ((main, 2), (crack(main[22], (main[22][0] + 70, main[22][1] - 46), 12, 2.0), 1),
                     (crack(main[41], (main[41][0] + 40, main[41][1] + 62), 10, 2.0), 1)):
        d.line(line, fill=(96, 88, 82), width=wd, joint="curve")
        md.line(line, fill=255, width=70)
    img = Image.composite(img.filter(ImageFilter.GaussianBlur(0.6)), img, mask)  # match the photo's softness
    return img, mask


def sink_v2(sink):
    """Sketch mold on the silicone joints and limescale on the faucet and around its base."""
    rng = np.random.default_rng(4)
    img = sink.copy()
    size = img.size
    mask = Image.new("L", size, 0)
    md = ImageDraw.Draw(mask)
    joints = [[(398, 417), (550, 440), (640, 452)], [(762, 440), (900, 458), (1024, 477)],
              [(792, 484), (900, 502), (1024, 528)]]
    back = [(0, 414), (140, 414), (282, 413)]  # the joint in the out-of-focus background
    # black mold: dense speckles hugging the joint
    pts = [p for j in joints for p in along(j)]
    mold = blotches(size, rng, pts, 3.2, 900, (2.5, 1.6)).filter(ImageFilter.GaussianBlur(0.9))
    img = Image.composite(Image.new("RGB", size, (38, 36, 32)), img, mold.point(lambda v: int(v * 0.85)))
    halo = blotches(size, rng, pts, 7, 500, (5, 3.5)).filter(ImageFilter.GaussianBlur(3))
    img = Image.composite(Image.new("RGB", size, (96, 88, 74)), img, halo.point(lambda v: int(v * 0.35)))
    bm = blotches(size, rng, along(back), 4, 220, (4, 2)).filter(ImageFilter.GaussianBlur(3.5))
    img = Image.composite(Image.new("RGB", size, (60, 58, 52)), img, bm.point(lambda v: int(v * 0.7)))
    for j in joints:
        md.line(j, fill=255, width=34, joint="curve")
    md.line(back, fill=255, width=30)
    # limescale: chalky crust around the base, streaks on the body and the spout end, a stain ring on the stone
    ring = along([(612, 492), (660, 506), (720, 508), (770, 498), (800, 484)])
    stain = blotches(size, rng, ring, 13, 120, (14, 6)).filter(ImageFilter.GaussianBlur(5))
    img = Image.composite(Image.new("RGB", size, (150, 138, 118)), img, stain.point(lambda v: int(v * 0.4)))
    crust = blotches(size, rng, ring + along([(690, 455), (770, 455)]), 4.5, 420, (7, 4)).filter(ImageFilter.GaussianBlur(1.2))
    img = Image.composite(Image.new("RGB", size, (232, 230, 220)), img, crust.point(lambda v: int(v * 0.8)))
    body = along([(700, 300), (712, 440)]) + along([(520, 380), (560, 392)]) + along([(690, 250), (745, 262)])
    scale = blotches(size, rng, body, 5, 520, (16, 22)).filter(ImageFilter.GaussianBlur(1.6))
    fa = Image.new("L", size, 0)  # limit the body speckles to the faucet silhouette (darker than the tiles)
    g = np.asarray(sink.convert("L")).astype(np.float32)
    fa_np = ((g < 150) * 255).astype(np.uint8)
    fa_np[:, :500] = 0
    fa_np[:, 800:] = 0
    fa_np[:180] = 0
    fa_np[500:] = 0
    fa = Image.fromarray(fa_np).filter(ImageFilter.GaussianBlur(1.5))
    scale = Image.fromarray((np.asarray(scale).astype(np.float32) * np.asarray(fa) / 255).astype(np.uint8))
    img = Image.composite(Image.new("RGB", size, (214, 212, 200)), img, scale.point(lambda v: int(v * 0.75)))
    md.rectangle((500, 185, 795, 470), fill=255)  # faucet
    md.ellipse((585, 448, 835, 530), fill=255)  # base and the stone around it
    return img, mask


def touch_up():
    """Third round: the v2 befores were good (inpaint strength 0.55), but the model had cleaned away the screw
    holes and most of the mold. Sketch just those again on the chosen v2 images and repaint only small spots
    at a low strength (inputs/*_touch.png, masks/*_touch.png)."""
    wall_p = ROOT / "outputs/wall_before_v2/wall_before_v2_s5.png"
    sink_p = ROOT / "outputs/sink_before_v2/sink_before_v2_s5.png"
    if not (wall_p.exists() and sink_p.exists()):
        return
    rng = np.random.default_rng(7)
    img = Image.open(wall_p).convert("RGB")
    mask = Image.new("L", img.size, 0)
    d, md = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    for x, y in [(262, 392), (300, 394), (455, 396), (640, 392), (668, 390), (478, 38)]:
        d.ellipse((x - 6, y - 5, x + 5, y + 6), fill=(214, 209, 201))  # chipped paint around the hole
        d.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=(52, 46, 42))
        d.ellipse((x - 1, y - 2.5, x + 2.5, y + 1), fill=(30, 26, 24))
        md.ellipse((x - 26, y - 26, x + 26, y + 26), fill=255)
    img = Image.composite(img.filter(ImageFilter.GaussianBlur(0.5)), img, mask)
    img.save(ROOT / "inputs/wall_touch.png")
    mask.save(ROOT / "masks/wall_touch.png")

    img = Image.open(sink_p).convert("RGB")
    size = img.size
    mask = Image.new("L", size, 0)
    md = ImageDraw.Draw(mask)
    joints = [[(398, 418), (550, 441), (642, 453)], [(790, 484), (900, 503), (1024, 529)],
              [(762, 441), (900, 459), (1024, 478)]]
    for j, (n_dark, n_tint) in zip(joints, ((520, 260), (620, 300), (160, 120))):
        pts = along(j, step=2)
        # uneven: a few dense colonies and sparse specks in between
        centres = rng.choice(len(pts), 9)
        colony = [pts[int(np.clip(c + rng.normal(0, 14), 0, len(pts) - 1))] for c in centres for _ in range(30)]
        tint = blotches(size, rng, pts, 5, n_tint, (3, 2.2)).filter(ImageFilter.GaussianBlur(2.2))
        img = Image.composite(Image.new("RGB", size, (112, 104, 86)), img, tint.point(lambda v: int(v * 0.4)))
        dark = blotches(size, rng, colony + pts[::6], 1.9, n_dark, (1.6, 1.1)).filter(ImageFilter.GaussianBlur(0.7))
        img = Image.composite(Image.new("RGB", size, (30, 29, 26)), img, dark.point(lambda v: int(v * 0.9)))
        md.line(j, fill=255, width=22, joint="curve")
    img.save(ROOT / "inputs/sink_touch.png")
    mask.save(ROOT / "masks/sink_touch.png")


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
    ws, wm2 = wall_v2(wall)
    ss, sm2 = sink_v2(sink)
    ws.save(ROOT / "inputs/wall_sketch.png")
    wm2.save(ROOT / "masks/wall_v2.png")
    ss.save(ROOT / "inputs/sink_sketch.png")
    sm2.save(ROOT / "masks/sink_v2.png")
    touch_up()
    if "--preview" in sys.argv:
        red = Image.new("RGB", sink.size, (255, 0, 0))
        Image.blend(sink, Image.composite(red, sink, sm), 0.45).save(ROOT / "preview_sink.png")
        Image.blend(wall, Image.composite(red, wall, wm), 0.45).save(ROOT / "preview_wall.png")
    print("masks written")


if __name__ == "__main__":
    main()

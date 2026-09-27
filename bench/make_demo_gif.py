#!/usr/bin/env python3
"""Render bench/demo_output.txt as a typing-animation GIF."""
from PIL import Image, ImageDraw, ImageFont

W, H, PAD, LH = 860, 460, 18, 24
BG, FG, GREEN, DIM = (18, 18, 22), (230, 230, 230), (120, 220, 130), (150, 150, 160)

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
except OSError:
    FONT = ImageFont.load_default()

raw = open("bench/demo_output.txt").read().splitlines()


def wrap(line):
    out, cur = [], ""
    for word in line.split(" "):
        t = (cur + " " + word).strip()
        d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        if d.textlength(t, font=FONT) > W - 2 * PAD and cur:
            out.append(cur)
            cur = word
        else:
            cur = t
    out.append(cur)
    return out


lines = []
for ln in raw:
    lines.extend(wrap(ln) if ln.strip() else [""])


def render(n):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines[:n]):
        c = GREEN if ln.startswith("$") else (DIM if ln.startswith("[tinyinfer") else FG)
        d.text((PAD, PAD + i * LH), ln, font=FONT, fill=c)
    return im


frames = [render(0)]
for n in range(1, len(lines) + 1):
    frames.append(render(n))
    if lines[n - 1].startswith("$"):
        frames.extend([render(n)] * 4)  # pause on commands
frames.extend([frames[-1]] * 12)  # hold final frame
frames[0].save("bench/demo.gif", save_all=True, append_images=frames[1:],
               duration=90, loop=0)
print("wrote bench/demo.gif", len(frames), "frames")

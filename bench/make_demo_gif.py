#!/usr/bin/env python3
"""Phase 8: demo GIF. Renders CAPTURED real tinyinfer output as a terminal
typing animation (PIL frames -> ffmpeg GIF). Content is real; presentation
is synthesized."""
import subprocess
from PIL import Image, ImageDraw, ImageFont

W, H = 860, 480
BG, FG, GREEN, DIM = (18, 18, 24), (230, 230, 230), (120, 220, 130), (130, 130, 150)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
PAD, LH = 18, 24
FPS = 12

cmd = "$ ./build/tinyinfer prompt \"Once upon a time\" --max-tokens 80 --temperature 0.7 --seed 7"
with open("bench/demo_output.txt") as f:
    out_text = f.read().strip()

def render(lines, cursor=False):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = PAD
    for text, color in lines:
        # simple word wrap
        words, cur = text.split(" "), ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if d.textlength(t, font=FONT) > W - 2 * PAD and cur:
                d.text((PAD, y), cur, font=FONT, fill=color); y += LH; cur = wd
            else:
                cur = t
        if cur:
            d.text((PAD, y), cur, font=FONT, fill=color); y += LH
    if cursor:
        d.rectangle([PAD, y + 2, PAD + 10, y + 18], fill=FG)
    return img

frames = []
shown = [("tinyinfer — from-scratch Llama inference in C++", DIM)]

# type the command
for i in range(1, len(cmd) + 1):
    frames.append(render(shown + [(cmd[:i], FG)], cursor=True))
# pause, then "run"
frames += [render(shown + [(cmd, FG)])] * 6
# stream the output word by word
words = out_text.split(" ")
shown2 = shown + [(cmd, FG)]
acc = ""
for i, wd in enumerate(words):
    acc = (acc + " " + wd).strip()
    if i % 3 == 0 or i == len(words) - 1:
        frames.append(render(shown2 + [(acc, GREEN)]))
frames += [render(shown2 + [(acc, GREEN)])] * 24  # hold

frames[0].save("/tmp/demo_frames.gif", save_all=True, append_images=frames[1:],
                duration=1000 // FPS, loop=0)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "/tmp/demo_frames.gif",
                "-vf", "scale=860:-1", "bench/demo.gif"], check=True)
print(f"wrote bench/demo.gif ({len(frames)} frames)")

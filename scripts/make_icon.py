"""Draw Kin's icon: a dark robot screen with two glowing green smiling eyes. Writes assets/kin-icon.png.

Composed for a circular crop (Telegram, Devpost). Tweak the constants and re-run.

usage: python scripts/make_icon.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

SIZE = 2048           # drawn large, saved at 1024 for smooth edges
OUT = 1024
EYE_TOP = (120, 255, 170)     # light green
EYE_BOTTOM = (20, 215, 120)   # deeper green
GLOW = (40, 255, 150)
BODY = (30, 32, 38)                          # his dark grey shell, outside the screen
BG_CENTER, BG_EDGE = (12, 40, 34), (2, 7, 8)  # the screen: faint green lit from the eyes
SCREEN_W, SCREEN_H, SCREEN_RADIUS = 1640, 1060, 260
BEZEL = (58, 62, 72)

EYE_W, EYE_H, EYE_RADIUS = 600, 540, 175
EYE_GAP, EYE_Y = 400, 1035    # half the distance between eye centers; vertical center
TILT = 5                      # degrees; inner corners raised, which reads as friendly rather than stern


def radial_background() -> Image.Image:
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    d = np.clip(np.hypot(x - SIZE / 2, y - SIZE * 0.48) / (SIZE * 0.62), 0, 1)[..., None]
    pixels = np.array(BG_CENTER) * (1 - d) + np.array(BG_EDGE) * d
    return Image.fromarray(pixels.astype(np.uint8), "RGB")


def eye_mask(cx: int, tilt: float) -> Image.Image:
    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)
    box = (cx - EYE_W // 2, EYE_Y - EYE_H // 2, cx + EYE_W // 2, EYE_Y + EYE_H // 2)
    draw.rounded_rectangle(box, radius=EYE_RADIUS, fill=255)
    # A happy eye is the top arch of the shape: carve the underside away with a wide ellipse.
    draw.ellipse((cx - EYE_W * 0.70, EYE_Y - 10, cx + EYE_W * 0.70, EYE_Y - 10 + 900), fill=0)
    return mask.rotate(tilt, resample=Image.BICUBIC, center=(cx, EYE_Y))


def screen_mask() -> Image.Image:
    mask = Image.new("L", (SIZE, SIZE), 0)
    box = ((SIZE - SCREEN_W) // 2, (SIZE - SCREEN_H) // 2, (SIZE + SCREEN_W) // 2, (SIZE + SCREEN_H) // 2)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=SCREEN_RADIUS, fill=255)
    return mask


def main() -> None:
    center = SIZE // 2
    eyes = ImageChops.lighter(eye_mask(center - EYE_GAP, TILT), eye_mask(center + EYE_GAP, -TILT))

    # Green fill, lighter at the top, with the faint scanlines of his LCD face.
    ramp = np.linspace(0, 1, SIZE)[:, None, None]
    fill = np.array(EYE_TOP) * (1 - ramp) + np.array(EYE_BOTTOM) * ramp
    fill = np.broadcast_to(fill, (SIZE, SIZE, 3)).copy()
    fill[(np.arange(SIZE) // 14) % 2 == 1] *= 0.86
    eye_layer = Image.fromarray(fill.astype(np.uint8), "RGB")

    screen = radial_background()
    for radius, strength in ((150, 0.45), (55, 0.55)):  # wide soft halo, then a tighter one
        halo = eyes.filter(ImageFilter.GaussianBlur(radius)).point(lambda v: int(v * strength))
        screen.paste(Image.new("RGB", (SIZE, SIZE), GLOW), mask=halo)
    screen.paste(eye_layer, mask=eyes)

    # The screen sits in his dark shell, with a thin lighter bezel around it.
    image = Image.new("RGB", (SIZE, SIZE), BODY)
    glass = screen_mask()
    image.paste(Image.new("RGB", (SIZE, SIZE), BEZEL), mask=glass.filter(ImageFilter.MaxFilter(45)))
    image.paste(screen, mask=glass)

    out = Path(__file__).resolve().parent.parent / "assets" / "kin-icon.png"
    out.parent.mkdir(exist_ok=True)
    image.resize((OUT, OUT), Image.LANCZOS).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

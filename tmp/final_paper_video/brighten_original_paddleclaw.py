from pathlib import Path

from PIL import Image, ImageSequence


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "paddleclaw_big_claw_wave_cutout.gif"
TARGET = HERE / "paddleclaw_big_claw_wave_cutout_bright.gif"


def brighten(frame: Image.Image) -> Image.Image:
    rgba = frame.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    lut = [min(255, round(255 * ((value / 255) ** (1 / 1.16)) + 255 * 0.025)) for value in range(256)]
    result = Image.merge("RGBA", (red.point(lut), green.point(lut), blue.point(lut), alpha))
    return result


with Image.open(SOURCE) as image:
    frames = [brighten(frame) for frame in ImageSequence.Iterator(image)]
    durations = [frame.info.get("duration", image.info.get("duration", 80)) for frame in ImageSequence.Iterator(image)]

frames[0].save(
    TARGET,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=image.info.get("loop", 0),
    disposal=2,
    optimize=False,
)

print(TARGET)

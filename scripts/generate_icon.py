from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for candidate in (
        windows / "Fonts" / "seguisb.ttf",
        windows / "Fonts" / "segoeuib.ttf",
        windows / "Fonts" / "arialbd.ttf",
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def generate_icon(output: Path) -> None:
    canvas_size = 1024
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    inset = 52
    draw.rounded_rectangle(
        (inset, inset, canvas_size - inset, canvas_size - inset),
        radius=150,
        fill="#38d39f",
    )
    font = _font(390)
    bounds = draw.textbbox((0, 0), "BL", font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    position = ((canvas_size - width) / 2, (canvas_size - height) / 2 - bounds[1] - 18)
    draw.text(position, "BL", font=font, fill="#07120e")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate_icon(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

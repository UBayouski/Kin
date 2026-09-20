"""VLM smoke test without the robot: describe an image file."""

import sys

from PIL import Image

from kin import llm

if len(sys.argv) != 2:
    sys.exit("usage: python scripts/02_describe_image.py <image>")

print(llm.describe_image(Image.open(sys.argv[1]), "Describe what you see in two sentences."))

"""Ask a locally cached Qwen3-VL model one question about one image.

Example::

    python -m d4rt_agent.qwen3vl_image_test

The default image is the trajectory plot supplied for this probe. ``--image``
also accepts a local path, which is preferable on an offline compute node.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from .simple_v2 import DEFAULT_QWEN_MODEL, OfflineQwen


DEFAULT_IMAGE = (
    "https://us1.discourse-cdn.com/flex002/uploads/python1/original/"
    "3X/7/3/73e8fd433b3ae29464cf95ac4959016a79786fda.png"
)
DEFAULT_QUESTION = "Describe the trajectory in this image."


def load_image(source: str) -> Image.Image:
    """Load an RGB image from a local path or an HTTP(S) URL."""

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "Open-d4rt-Qwen3VL-test/1.0"})
        with urlopen(request, timeout=30) as response:
            payload = response.read()
        image_source: str | BytesIO = BytesIO(payload)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        image_source = str(path)

    with Image.open(image_source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Local image path or HTTP(S) URL (default: supplied trajectory image).",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help=f"Question sent with the image (default: {DEFAULT_QUESTION!r}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_QWEN_MODEL,
        help=f"Cached Hugging Face model ID or local model path (default: {DEFAULT_QWEN_MODEL}).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    image = load_image(args.image)
    qwen = OfflineQwen(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    answer = qwen.generate(messages)

    print(f"Model: {qwen.model_path}")
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print("Answer:")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

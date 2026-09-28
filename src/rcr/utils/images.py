"""Image paths shared by the annotation UIs and candidate reranker."""

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def image_relative_path(url: str) -> Path:
    """Map PIPA/local-files URLs to safe split-relative image paths."""

    parsed = urlparse(url)
    value = parse_qs(parsed.query).get("d", [parsed.path])[0]
    value = unquote(value).replace("\\", "/")
    for marker in ("/PIPA/images/", "/images/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = value.lstrip("/")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image path {url!r}")
    return path

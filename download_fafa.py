"""Resume a FAFA image cache download from a Kaggle notebook version."""

import argparse
import json
import re
from functools import partial
from pathlib import Path, PurePosixPath
from time import sleep

PAGE_SIZE = 200
RETRIES = 5


def cache_pattern(value):
    """Match one export root and only the known image-cache files beneath it."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("cache path must be a non-empty relative export directory")
    return re.compile(
        r"^(?:referential-composition-retrieval/)?"
        + re.escape(path.as_posix())
        + r"/(?P<file>index\.pt|targets/[0-9]+\.pt|\.persons/[0-9]+\.pt)$"
    )


def retry(call, label):
    for attempt in range(RETRIES):
        try:
            return call()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == RETRIES - 1:
                raise
            delay = min(5 * (attempt + 1), 30)
            print(f"{label}: {exc}; retry in {delay}s", flush=True)
            sleep(delay)


def download(session, url, target):
    part = target.with_name(target.name + ".part")

    def attempt():
        part.unlink(missing_ok=True)
        with session.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            header = response.headers.get("Content-Length")
            expected = int(header) if header and header.isdigit() else None
            size = 0
            with part.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        size += len(chunk)
            if not size or (expected is not None and size != expected):
                raise OSError(f"Incomplete download: {target.name} ({size}/{expected})")
            part.replace(target)

    retry(attempt, f"Download {target.name}")


def save_state(path, source, token, pages, downloaded, skipped):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "source": source,
                "token": token,
                "pages": pages,
                "downloaded": downloaded,
                "skipped": skipped,
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_targets(root):
    index_path = root / "index.pt"
    if not index_path.is_file():
        print("WARNING: index.pt is missing; cannot verify target count.")
        return
    try:
        import torch
    except ImportError:
        print("Install torch to verify target count")
        return
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    count = len(index["image_ids"])
    missing = [
        index
        for index in range(count)
        if not (root / "targets" / f"{index}.pt").is_file()
        or (root / "targets" / f"{index}.pt").stat().st_size == 0
    ]
    print(f"Targets present: {count - len(missing)}/{count}")
    print(f"Missing targets (first 20): {missing[:20]}")
    print("Note: existing files were not checksum-validated.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--user", required=True, help="Kaggle notebook owner")
    parser.add_argument("--kernel", required=True, help="Kaggle notebook slug")
    parser.add_argument("--version", required=True, help="Kaggle notebook version")
    parser.add_argument(
        "--cache-path",
        default="cache/person-images",
        help="Image-cache directory inside the notebook export (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        cache_pattern(args.cache_path)
    except ValueError as error:
        parser.error(str(error))
    args.cache_path = PurePosixPath(args.cache_path).as_posix()
    return args


def main(argv=None):
    args = parse_args(argv)
    # Keep argument parsing (especially --help) usable before optional FAFA
    # dependencies are installed.
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "user": args.user,
        "kernel": args.kernel,
        "version": args.version,
        "cache_path": args.cache_path,
    }
    state_path = root / ".download-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    saved_source = state.get("source", {})
    if saved_source and "cache_path" not in saved_source:
        # Resume manifests predating --cache-path always used this export root.
        saved_source = {**saved_source, "cache_path": "cache/fafa-images"}
    if state and saved_source != source:
        raise ValueError(
            "download state belongs to another Kaggle notebook/version/cache path; "
            f"remove {state_path} or use its original arguments"
        )
    token = state.get("token")
    pages = state.get("pages", 0)
    downloaded = state.get("downloaded", 0)
    skipped = state.get("skipped", 0)
    pattern = cache_pattern(args.cache_path)

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client, requests.Session() as session:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = args.user
            request.kernel_slug = args.kernel
            request.version_label = api._kernel_version_label(args.version)
            api._set_paging(request, PAGE_SIZE, token)
            response = retry(
                partial(
                    client.kernels.kernels_api_client.list_kernel_session_output,
                    request,
                ),
                "Kaggle file listing",
            )
            items = response.files or []
            matched = 0
            for item in items:
                match = pattern.fullmatch(item.file_name)
                if not match:
                    continue
                matched += 1
                target = root / match.group("file")
                if target.is_file() and target.stat().st_size > 0:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                download(session, item.url, target)
                downloaded += 1
                print(f"Downloaded: {target.relative_to(root)}", flush=True)

            next_token = response.next_page_token or None
            pages += 1
            print(
                f"Page {pages}: listed={len(items)}, matched={matched}, "
                f"downloaded={downloaded}, skipped={skipped}, "
                f"first={items[0].file_name if items else '-'}, "
                f"last={items[-1].file_name if items else '-'}",
                flush=True,
            )
            if next_token == token and next_token:
                raise RuntimeError("Kaggle returned the same page token twice")
            save_state(state_path, source, next_token, pages, downloaded, skipped)
            token = next_token
            if not token:
                break

    state_path.unlink(missing_ok=True)
    print(
        f"Finished version {args.version}: downloaded={downloaded}, skipped={skipped}"
    )
    verify_targets(root)


if __name__ == "__main__":
    main()

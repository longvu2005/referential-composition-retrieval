
"""Resume FAFA image cache from Kaggle notebook version 43."""

import json
import re
from pathlib import Path
from time import sleep

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)

ROOT = Path(
    "/Volumes/T9/referential-composition-retrieval-output/"
    "referential-composition-retrieval/cache/fafa-images"
)
STATE = ROOT / ".download-v43-state.json"

PATTERN = re.compile(
    r"^(?:referential-composition-retrieval/)?cache/fafa-images/"
    r"(?P<file>index\.pt|targets/[0-9]+\.pt|\.persons/[0-9]+\.pt)$"
)

PAGE_SIZE = 200
RETRIES = 5


def retry(call, label):
    for attempt in range(RETRIES):
        try:
            return call()
        except Exception as exc:
            status = getattr(
                getattr(exc, "response", None),
                "status_code",
                None,
            )
            if status is not None and status not in (
                408, 429, 500, 502, 503, 504
            ):
                raise

            if attempt == RETRIES - 1:
                raise

            delay = min(5 * (attempt + 1), 30)
            print(
                f"{label}: {exc}; retry in {delay}s",
                flush=True,
            )
            sleep(delay)


def download(session, url, target):
    part = target.with_name(target.name + ".part")

    def attempt():
        part.unlink(missing_ok=True)

        with session.get(
            url, stream=True, timeout=(20, 120)
        ) as response:
            response.raise_for_status()

            expected = response.headers.get("Content-Length")
            expected = (
                int(expected)
                if expected and expected.isdigit()
                else None
            )

            size = 0
            with part.open("wb") as stream:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        stream.write(chunk)
                        size += len(chunk)

            if not size or (
                expected is not None and size != expected
            ):
                raise IOError(
                    f"Incomplete download: {target.name} "
                    f"({size}/{expected})"
                )

            part.replace(target)

    retry(attempt, f"Download {target.name}")


def save_state(token, pages, downloaded, skipped):
    temporary = STATE.with_suffix(".tmp")

    temporary.write_text(
        json.dumps({
            "token": token,
            "pages": pages,
            "downloaded": downloaded,
            "skipped": skipped,
        }),
        encoding="utf-8",
    )
    temporary.replace(STATE)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    state = (
        json.loads(STATE.read_text(encoding="utf-8"))
        if STATE.exists()
        else {}
    )

    token = state.get("token")
    pages = state.get("pages", 0)
    downloaded = state.get("downloaded", 0)
    skipped = state.get("skipped", 0)

    api = KaggleApi()
    api.authenticate()

    with (
        api.build_kaggle_client() as client,
        requests.Session() as session,
    ):
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "phmhunhlongv"
            req.kernel_slug = "referential-composition-retrieval"
            req.version_label = api._kernel_version_label("43")

            api._set_paging(req, PAGE_SIZE, token)

            response = retry(
                lambda: client.kernels.kernels_api_client
                .list_kernel_session_output(req),
                "Kaggle file listing",
            )

            items = response.files or []
            matched = 0

            for item in items:
                match = PATTERN.fullmatch(item.file_name)
                if not match:
                    continue

                matched += 1
                target = ROOT / match.group("file")

                # Skip locally before any HTTP request.
                if (
                    target.is_file()
                    and target.stat().st_size > 0
                ):
                    skipped += 1
                    continue

                target.parent.mkdir(
                    parents=True, exist_ok=True
                )

                download(session, item.url, target)
                downloaded += 1

                print(
                    f"Downloaded: {target.relative_to(ROOT)}",
                    flush=True,
                )

            next_token = response.next_page_token or None
            pages += 1

            print(
                f"Page {pages}: "
                f"listed={len(items)}, "
                f"matched={matched}, "
                f"downloaded={downloaded}, "
                f"skipped={skipped}, "
                f"first={items[0].file_name if items else '-'}, "
                f"last={items[-1].file_name if items else '-'}",
                flush=True,
            )

            if next_token == token and next_token:
                raise RuntimeError(
                    "Kaggle returned the same page token twice"
                )

            save_state(
                next_token, pages, downloaded, skipped
            )

            token = next_token
            if not token:
                break

    STATE.unlink(missing_ok=True)

    print(
        f"Finished version 43: "
        f"downloaded={downloaded}, skipped={skipped}"
    )

    index_path = ROOT / "index.pt"

    if index_path.is_file():
        try:
            import torch
        except ImportError:
            print("Install torch to verify target count")
        else:
            index = torch.load(
                index_path,
                map_location="cpu",
                weights_only=True,
            )

            count = len(index["image_ids"])
            missing = [
                i for i in range(count)
                if not (
                    ROOT / "targets" / f"{i}.pt"
                ).is_file()
                or (
                    ROOT / "targets" / f"{i}.pt"
                ).stat().st_size == 0
            ]

            print(
                f"Targets present: "
                f"{count - len(missing)}/{count}"
            )
            print(
                f"Missing targets (first 20): {missing[:20]}"
            )
            print(
                "Note: existing files were not "
                "checksum-validated."
            )
    else:
        print(
            "WARNING: index.pt is missing; "
            "cannot verify target count."
        )


if __name__ == "__main__":
    main()

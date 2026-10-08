"""Export comparable RCR paper tables from saved results, without inference."""

import argparse
from pathlib import Path

from rcr.evaluation.report import METHODS, ReportError, default_runs, generate_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reports"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD=DIRECTORY",
        help="Override one method's existing split result directory",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("csv", "md", "tex"),
        default=["csv", "md", "tex"],
    )
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args(argv)
    directories = default_runs(args.runs_root, args.split)
    seen = set()
    for value in args.run:
        name, separator, path = value.partition("=")
        if not separator or name not in args.methods or not path or name in seen:
            parser.error("--run requires a unique selected METHOD=DIRECTORY")
        directories[name] = Path(path)
        seen.add(name)
    try:
        paths = generate_report(
            {name: directories[name] for name in dict.fromkeys(args.methods)},
            split=args.split,
            output_dir=args.output_dir,
            formats=args.formats,
            decimals=args.decimals,
        )
    except ReportError as error:
        parser.error(str(error))
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()

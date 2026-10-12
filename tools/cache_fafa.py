"""Isolated FAFA worker, normally invoked by the proposed experiment CLI."""

import argparse

from rcr.common.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.force and not args.prepare:
        parser.error("--force requires --prepare")
    cfg = load_config(args.config)
    if args.prepare:
        from rcr.baselines.prepare import prepare_fafa_model
        from rcr.proposed.cache.build import load_fafa_config

        prepare_fafa_model(load_fafa_config(cfg), force=args.force)
    else:
        from rcr.proposed.cache.fafa import finish_fafa_cache

        finish_fafa_cache(cfg)


if __name__ == "__main__":
    main()

"""Isolated FAFA worker, normally invoked by the proposed experiment CLI."""

import argparse

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.force and not args.prepare:
        parser.error("--force requires --prepare")
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.check:
        from rcr.methods.proposed.person_encoder import check_fafa_assets

        check_fafa_assets(cfg)
    elif args.prepare:
        from rcr.methods.baselines.prepare import prepare_fafa_model
        from rcr.methods.proposed.person_encoder import load_fafa_config

        prepare_fafa_model(load_fafa_config(cfg), force=args.force)
    else:
        from rcr.methods.proposed.fafa_cache import finish_fafa_cache

        finish_fafa_cache(cfg)


if __name__ == "__main__":
    main()

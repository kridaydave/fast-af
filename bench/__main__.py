"""CLI entry: python -m bench --config bench/config.yaml --out results.yaml"""
import argparse

from .run import run


def main():
    ap = argparse.ArgumentParser(description="EdgeFast-Diff 1B benchmark week")
    ap.add_argument("--config", default="bench/config.yaml")
    ap.add_argument("--out", default="results.yaml")
    ap.add_argument("--only", help="comma-separated model ids to run")
    args = ap.parse_args()

    cfg = __import__("yaml").safe_load(open(args.config).read())
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        cfg["models"] = [m for m in cfg["models"] if m["id"] in want]

    run(args.config, args.out)


if __name__ == "__main__":
    main()

"""Command line entry point for task construction and recorded-completion grading."""
import argparse
import json
import sys

from .pipeline import build, grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="generate and audit a new immutable task bundle")
    make.add_argument("--output", required=True)
    make.add_argument("--variants", type=int, default=2)
    make.add_argument("--seed", type=int, default=17)
    make.add_argument("--benchmarks", help="sources JSON containing pinned benchmark paths")
    make.add_argument("--tokenizer")
    make.add_argument("--tokenizer-manifest")
    make.add_argument("--max-tokens", type=int, default=1024)
    make.add_argument("--review-file", help="explicit catalog review attestation; never generated automatically")
    make.add_argument("--smoke", action="store_true", help="mark output as plumbing-only")
    check = commands.add_parser("grade", help="verify frozen-checkpoint completions")
    check.add_argument("--bundle", required=True)
    check.add_argument("--attempts", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--generation-seconds", type=float, required=True)
    check.add_argument("--gpu-seconds", type=float, required=True)
    check.add_argument("--split", choices=("train", "development"), default="train")
    check.add_argument("--min-accepted", type=int, default=1)
    check.add_argument("--min-successes-per-gpu-hour", type=float, default=0.0)
    check.add_argument("--previous-accepted", help="prior accepted examples to exclude from this round")
    args = vars(parser.parse_args())
    operation = args.pop("command")
    try:
        result = build(**args) if operation == "build" else grade(**args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

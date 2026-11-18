"""CLI for release validation and Motion Code compilation."""

from __future__ import annotations

import argparse
import json

from .compiler import compile_motion_code
from .release import validate_release


def main() -> None:
    parser = argparse.ArgumentParser(prog="motion-programs")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate curated paper data")
    validate.add_argument("root")
    compile_parser = sub.add_parser("compile", help="compile one Motion Code")
    compile_parser.add_argument("code")
    compile_parser.add_argument("--platform", required=True)
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_release(args.root)
    else:
        result = compile_motion_code(args.code, args.platform)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

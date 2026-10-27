#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUN_PLACEHOLDER = re.compile(r"/PATH/TO/out/([^/]+)/RUN/([^'\"\s,\]]+)")
RELATIVE_KEYS = re.compile(r"^(extends|output_dir):(\s*)(['\"]?)([^'\"\s#]+)(['\"]?)", re.MULTILINE)


def resolve_weight(dpc: Path, training: str, file_name: str, chosen: dict[str, Path]) -> Path:
    if training in chosen:
        path = chosen[training]
        path = path / file_name if path.is_dir() else path
        if not path.is_file():
            raise SystemExit(f"Weights given for '{training}' do not exist: {path}")
        return path
    root = dpc / "out" / training
    candidates = sorted(
        path
        for path in (root / file_name, *root.glob(f"*/{file_name}"))
        if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No {file_name} of '{training}' under {root}")
    listed = "\n  ".join(str(path) for path in candidates)
    raise SystemExit(
        f"Several {file_name} of '{training}', choose one with "
        f"--weight {training}=<file>:\n  {listed}"
    )


def fill(
    text: str,
    template: Path,
    exp: Path | None,
    dpc: Path | None,
    chosen: dict[str, Path] | None = None,
) -> str:
    if dpc is not None:
        text = RUN_PLACEHOLDER.sub(
            lambda match: str(resolve_weight(dpc, match.group(1), match.group(2), chosen or {})),
            text,
        )
        text = text.replace("/PATH/TO/Pano3D", str(dpc))
    if exp is not None:
        text = text.replace("/PATH/TO/front3d", str(exp))

    def absolute(match: re.Match[str]) -> str:
        key, space, quote, value, closing = match.groups()
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (template.parent / path).resolve()
        return f"{key}:{space}{quote}{path}{closing}"

    text = RELATIVE_KEYS.sub(absolute, text)
    left = sorted(set(re.findall(r"/PATH/TO[^'\"\s,\]]*", text)))
    if left:
        raise SystemExit(f"{template}: placeholders left unfilled: {', '.join(left)}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="fill the /PATH/TO placeholders of a config")
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exp", type=Path, help="3D-FRONT experiment folder")
    parser.add_argument("--dpc", type=Path, help="DeepPanoContext clone")
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="TRAINING=PATH",
        help="weights for /PATH/TO/out/TRAINING/RUN/: a file or a run folder",
    )
    args = parser.parse_args()
    chosen = {}
    for item in args.weight:
        training, separator, path = item.partition("=")
        if not separator:
            raise SystemExit(f"--weight expects TRAINING=PATH, got '{item}'")
        chosen[training] = Path(path).expanduser()
    template = args.template.resolve()
    text = fill(template.read_text(encoding="utf-8"), template, args.exp, args.dpc, chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

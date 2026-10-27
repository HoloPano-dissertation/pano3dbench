#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUN_PLACEHOLDER = re.compile(r"/PATH/TO/out/([^/]+)/RUN/([^'\"\s,\]]+)")
RELATIVE_KEYS = re.compile(r"^(extends|output_dir):(\s*)(['\"]?)([^'\"\s#]+)(['\"]?)", re.MULTILINE)


def latest_run(dpc: Path, training: str, file_name: str) -> Path:
    runs = sorted(
        path.parent
        for path in (dpc / "out" / training).glob(f"*/{file_name}")
        if path.is_file()
    )
    if not runs:
        raise SystemExit(f"No run of '{training}' with {file_name} under {dpc / 'out'}")
    return runs[-1]


def fill(text: str, template: Path, exp: Path | None, dpc: Path | None) -> str:
    if dpc is not None:
        text = RUN_PLACEHOLDER.sub(
            lambda match: str(latest_run(dpc, match.group(1), match.group(2)) / match.group(2)),
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
    args = parser.parse_args()
    template = args.template.resolve()
    text = fill(template.read_text(encoding="utf-8"), template, args.exp, args.dpc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

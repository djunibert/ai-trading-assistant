import argparse
from pathlib import Path


def read_requirements(path: Path) -> list[str]:
    data = path.read_bytes()

    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")

    return text.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="requirements.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-package", action="append", default=[])
    args = parser.parse_args()

    excluded = tuple(package.lower() + "==" for package in args.exclude_package)
    lines = [
        line
        for line in read_requirements(Path(args.input))
        if not line.lower().startswith(excluded)
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

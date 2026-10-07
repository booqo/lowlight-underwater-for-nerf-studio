#!/usr/bin/env python3
"""Export the current source tree without Git history or ignored local files."""

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def source_files(include_untracked=False):
    command = ["git", "ls-files", "-z", "--cached"]
    if include_untracked:
        command += ["--others", "--exclude-standard"]
    raw = subprocess.check_output(command, cwd=ROOT)
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        input=raw, cwd=ROOT, stdout=subprocess.PIPE, check=False,
    )
    if ignored.returncode not in (0, 1):
        raise RuntimeError("Could not determine ignored files")
    excluded = set(ignored.stdout.decode().split("\0"))
    for name in sorted(set(raw.decode().split("\0")) - excluded - {""}):
        path = ROOT / name
        if path.is_symlink():
            raise ValueError(f"Refusing to export symlink: {name}")
        if path.is_file():
            yield name, path


def export(output, include_untracked=False):
    files = list(source_files(include_untracked))
    if not files:
        raise ValueError("No source files selected")
    if any(path.resolve() == output.resolve() for _, path in files):
        raise ValueError("Output path must not overwrite a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Exclusive creation protects existing archives; metadata omits local user names.
    with output.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, path in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(f"llsu-source/{name}")
                    info.size = len(data)
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    archive.addfile(info, io.BytesIO(data))
                    manifest.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
                data = "".join(manifest).encode()
                info = tarfile.TarInfo("llsu-source/RELEASE_MANIFEST.sha256")
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    print(f"Exported {len(files)} source files to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-untracked", action="store_true",
                        help="Also include reviewed, non-ignored new files")
    args = parser.parse_args()
    export(args.output, args.include_untracked)

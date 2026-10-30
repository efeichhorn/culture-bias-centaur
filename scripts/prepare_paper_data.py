#!/usr/bin/env python3
"""Check and extract the saved study data. No downloads or model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def stream_sha256(stream) -> str:
    """Hash a file or archive member in small chunks."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha256(stream)


def member_path(name: str) -> Path:
    """Accept only normal relative file paths."""
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe archive path: {name}")
    return Path(*path.parts)


def target_path(output: Path, relative: Path) -> Path:
    """Reject symlinks so extraction cannot escape through an existing folder."""
    target = output / relative
    for path in [target, *target.parents]:
        if path.is_symlink():
            raise ValueError(f"Symlink in output path: {path}")
        if path == output:
            break
    if not target.resolve().is_relative_to(output):
        raise ValueError(f"Output path escapes destination: {target}")
    return target


def prepare(archive: Path, manifest_path: Path, output: Path, verify_only: bool = False) -> dict:
    """Verify every member before writing; keep existing matching files."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data manifest version")
    if archive.stat().st_size != manifest["archive_bytes"] or sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Archive differs from the supplied manifest")
    entries = manifest["entries"]
    expected = {entry["path"]: entry for entry in entries}
    if len(expected) != len(entries) or len(entries) != manifest["files"]:
        raise ValueError("Duplicate paths or incorrect file count in manifest")
    if sum(entry["bytes"] for entry in entries) != manifest["expanded_bytes"]:
        raise ValueError("Incorrect expanded size in manifest")
    # Resolve the parent, but reject a symlink at the requested destination.
    if output.is_symlink():
        raise ValueError("The output directory cannot be a symlink")
    output = output.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(expected) or {item.filename for item in members} != set(expected):
            raise ValueError("Archive members differ from the manifest")
        for item in members:
            relative = member_path(item.filename)
            if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Expected a regular file: {item.filename}")
            entry = expected[item.filename]
            if item.file_size != entry["bytes"]:
                raise ValueError(f"Incorrect size: {item.filename}")
            with bundle.open(item) as stream:
                digest = stream_sha256(stream)
            if digest != entry["sha256"]:
                raise ValueError(f"Incorrect file hash: {item.filename}")
            if not verify_only:
                target = target_path(output, relative)
                if target.exists() and (not target.is_file() or sha256(target) != digest):
                    raise ValueError(f"Existing file differs; choose a new --output directory: {target}")
        written = 0
        if not verify_only:
            for item in members:
                target = target_path(output, member_path(item.filename))
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        with bundle.open(item) as source:
                            shutil.copyfileobj(source, stream)
                    if sha256(temporary) != expected[item.filename]["sha256"]:
                        raise ValueError(f"Extracted file hash differs: {item.filename}")
                    # A hard link installs the complete file without replacing one.
                    os.link(temporary, target)
                    written += 1
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    return {"verified_files": len(entries), "written_files": written,
            "expanded_bytes": manifest["expanded_bytes"], "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/paper_inputs.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/paper_inputs.manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/workspace")
    parser.add_argument("--verify-only", action="store_true", help="Check the archive without extracting it")
    args = parser.parse_args()
    result = prepare(args.archive, args.manifest, args.output, args.verify_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

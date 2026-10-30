import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/prepare_paper_data.py"
spec = importlib.util.spec_from_file_location("prepare_paper_data", MODULE)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def bundle(tmp_path, files):
    archive = tmp_path / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, content in files.items():
            handle.writestr(name, content)
    entries = [{"path": name, "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}
               for name, content in files.items()]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "archive_bytes": archive.stat().st_size,
        "archive_sha256": data.sha256(archive), "files": len(entries),
        "expanded_bytes": sum(entry["bytes"] for entry in entries), "entries": entries,
    }))
    return archive, manifest


def test_extract_and_repeat(tmp_path):
    archive, manifest = bundle(tmp_path, {"data/rows.csv": b"item,value\na,1\n"})
    output = tmp_path / "output"
    assert data.prepare(archive, manifest, output)["written_files"] == 1
    assert (output / "data/rows.csv").read_bytes() == b"item,value\na,1\n"
    assert data.prepare(archive, manifest, output)["written_files"] == 0


def test_verify_only_does_not_write(tmp_path):
    archive, manifest = bundle(tmp_path, {"rows.csv": b"data"})
    output = tmp_path / "output"
    assert data.prepare(archive, manifest, output, verify_only=True)["verified_files"] == 1
    assert not output.exists()


def test_corrupted_member_does_not_write(tmp_path):
    archive, manifest = bundle(tmp_path, {"rows.csv": b"data"})
    record = json.loads(manifest.read_text())
    record["entries"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incorrect file hash"):
        data.prepare(archive, manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", ["../escape.csv", "/escape.csv", "folder/../escape.csv", "C:/escape.csv"])
def test_unsafe_member_rejected(tmp_path, name):
    archive, manifest = bundle(tmp_path, {name: b"data"})
    with pytest.raises(ValueError, match="Unsafe archive path"):
        data.prepare(archive, manifest, tmp_path / "output")
    assert not (tmp_path / "escape.csv").exists()


def test_existing_mismatch_stops_before_other_files_are_written(tmp_path):
    archive, manifest = bundle(tmp_path, {"new.csv": b"new", "existing.csv": b"expected"})
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.csv").write_bytes(b"keep me")
    with pytest.raises(ValueError, match="Existing file differs"):
        data.prepare(archive, manifest, output)
    assert (output / "existing.csv").read_bytes() == b"keep me"
    assert not (output / "new.csv").exists()


def test_output_symlink_cannot_escape(tmp_path):
    archive, manifest = bundle(tmp_path, {"folder/rows.csv": b"data"})
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "folder").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink in output path"):
        data.prepare(archive, manifest, output)
    assert not (outside / "rows.csv").exists()

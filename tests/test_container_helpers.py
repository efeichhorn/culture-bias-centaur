"""Cluster commands work with either supported container runtime."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("runtime", ["apptainer", "singularity"])
def test_container_command_uses_available_runtime(tmp_path, runtime):
    executable = tmp_path / runtime
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    command = 'source scripts/container_helpers.sh; container_python "image with spaces.sif" 1 script.py --flag value'
    environment = {**os.environ, "PATH": str(tmp_path), "REPO_ROOT": str(root)}
    for name in ("HF_HOME", "VIPER_PYTHON_OVERLAY"):
        environment.pop(name, None)
    result = subprocess.run(["/bin/bash", "-c", command], cwd=root,
                            env=environment, text=True, capture_output=True, check=True)
    arguments = result.stdout.splitlines()
    assert arguments[0] == "exec"
    assert "--rocm" in arguments
    assert arguments[-5:] == ["image with spaces.sif", "python", "script.py", "--flag", "value"]

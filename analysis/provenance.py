"""Read the verified dataset revision without requiring retained Git history."""
import json
from pathlib import Path
import subprocess


def dataset_revision(root):
    manifest = Path(root) / 'dataset_manifest.json'
    if manifest.exists():
        return json.loads(manifest.read_text())['revision']
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()

"""
Persistence layer for Bayesian ranking models.

Saves / loads the full HierarchicalModel state as JSON.
Keeps the last 5 backups automatically.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

from user_model import HierarchicalModel

DEFAULT_MODEL_DIR = Path("models")
MODEL_FILE = "hierarchical_model.json"


def save_model(model, path=None):
    """Persist a HierarchicalModel to disk (with auto-backup)."""
    model_dir = Path(path) if path else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    filepath = model_dir / MODEL_FILE

    # Rotate backups
    if filepath.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = model_dir / f"hierarchical_model_backup_{ts}.json"
        shutil.copy2(filepath, backup)
        for old in sorted(model_dir.glob("hierarchical_model_backup_*.json"))[:-5]:
            old.unlink()

    data = model.to_dict()
    data["_saved_at"] = datetime.now(timezone.utc).isoformat()

    # Atomic write: tmp → rename
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(filepath)


def load_model(path=None):
    """Load a HierarchicalModel from disk, or return a fresh one."""
    model_dir = Path(path) if path else DEFAULT_MODEL_DIR
    filepath = model_dir / MODEL_FILE

    if not filepath.exists():
        return HierarchicalModel()

    with open(filepath) as f:
        data = json.load(f)

    data.pop("_saved_at", None)
    return HierarchicalModel.from_dict(data)

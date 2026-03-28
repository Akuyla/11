import sys
from pathlib import Path


def _append_resnest_path():
    repo_root = Path(__file__).resolve().parent.parent
    resnest_root = repo_root / 'ResNeSt'
    resnest_path = str(resnest_root)
    if resnest_root.exists() and resnest_path not in sys.path:
        sys.path.insert(0, resnest_path)


_append_resnest_path()

from models.models import resnest50 as _resnest50  # noqa: E402


def build_resnest50(pretrained=False):
    return _resnest50(pretrained=pretrained)

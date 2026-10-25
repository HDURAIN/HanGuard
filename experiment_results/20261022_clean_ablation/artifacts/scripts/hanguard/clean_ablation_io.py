"""Block accidental reads of historical experiment artifacts in clean jobs."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


class HistoricalArtifactReadGuard:
    """A process-local audit hook; project implementation code remains readable.

    Model/runtime files and raw corpus files are not experiment results. Only
    the exact new study and its own result bundle are readable under the two
    repository experiment roots. No filesystem content is changed by this hook.
    """

    def __init__(self, study, archive, root=ROOT):
        self.root = Path(root).resolve()
        self.study = Path(study).resolve()
        self.archive = Path(archive).resolve()
        self.enabled = True
        sys.addaudithook(self._audit)

    def _audit(self, event, args):
        if not self.enabled or event not in ("open", "os.listdir", "os.scandir") or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(raw)).resolve()
        protected = (self.root / "outputs", self.root / "experiment_results")
        if not any(path.is_relative_to(folder) for folder in protected):
            return
        if path.is_relative_to(self.study) or path.is_relative_to(self.archive):
            return
        raise PermissionError(f"Clean ablation cannot read a historical experiment artifact: {path}")

    def close(self):
        self.enabled = False


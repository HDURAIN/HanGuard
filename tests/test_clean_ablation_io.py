"""Exercise the real audit hook without touching historical experiment files."""
from pathlib import Path
import tempfile
import unittest

from scripts.hanguard.clean_ablation_io import HistoricalArtifactReadGuard


class CleanArtifactReadGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "outputs/new"
        self.archive = self.root / "experiment_results/new"
        self.old = self.root / "outputs/old"
        for directory in (self.study, self.archive, self.old,
                          self.root / "data/raw", self.root / "models/base",
                          self.root / "experiment_results/old"):
            directory.mkdir(parents=True)
            (directory / "sample.json").write_text("fresh or historical fixture")
        self.guard = HistoricalArtifactReadGuard(self.study, self.archive, self.root)
        self.addCleanup(self.guard.close)

    def test_new_study_archive_raw_data_and_model_remain_readable(self):
        for directory in (self.study, self.archive, self.root / "data/raw", self.root / "models/base"):
            self.assertTrue((directory / "sample.json").read_text())

    def test_old_result_and_intermediate_reads_are_blocked(self):
        for directory in (self.old, self.root / "experiment_results/old"):
            with self.assertRaises(PermissionError):
                (directory / "sample.json").read_text()

    def test_historical_directory_enumeration_is_blocked(self):
        with self.assertRaises(PermissionError):
            list(self.old.iterdir())

    def test_symlink_cannot_import_historical_features(self):
        link = self.study / "looks_new.json"
        link.symlink_to(self.old / "sample.json")
        with self.assertRaises(PermissionError):
            link.read_text()

    def test_guard_can_be_disabled_for_test_cleanup(self):
        self.guard.close()
        self.assertTrue((self.old / "sample.json").read_text())


if __name__ == "__main__":
    unittest.main()

"""Synthetic publish-boundary checks. No real credentials or files."""

import importlib.util
import contextlib
import hashlib
import io
import sys
import zipfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_check", ROOT / "scripts/check_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, name, text="synthetic"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_allows_authored_source_and_demo_asset(self):
        files = [self.file("README.md"), self.file("src/chatlens/cli.py"), self.file("examples/demo.json", "{}")]
        self.assertEqual(release.scan_paths(self.root, files), [])

    def test_rejects_sensitive_runtime_files_even_inside_source_tree(self):
        files = [
            self.file("src/private/bridge-session.json"),
            self.file("examples/chat.sqlite3"),
            self.file("models/weights.safetensors"),
        ]
        categories = {f["category"] for f in release.scan_paths(self.root, files)}
        self.assertIn("private_or_generated_artifact", categories)
        self.assertIn("unapproved_file_type", categories)

    def test_reports_secret_category_and_line_without_matching_text(self):
        credential = "ghp_" + "x" * 40
        file = self.file("src/check.py", "# synthetic\n" + credential)
        findings = release.scan_paths(self.root, [file])
        self.assertEqual(findings, [{"path": "src/check.py", "category": "github_credential", "line": 2}])
        self.assertNotIn(credential, str(findings))

    def test_symlink_cannot_escape_source_allowlist(self):
        target = self.file("README.md")
        link = self.root / "source-link"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("Symlinks unavailable on this runner")
        self.assertEqual(release.scan_paths(self.root, [link])[0]["category"], "symlink_not_publishable")

    def test_forced_tracked_ignored_file_is_included(self):
        file = self.file("build/session.json")
        result = mock.Mock(returncode=0, stdout=b"build/session.json\0")
        with mock.patch.object(release.subprocess, "run", return_value=result):
            files = release.candidate_files(self.root)
        self.assertIn(file, files)
        self.assertTrue(release.scan_paths(self.root, files))

    def test_local_deny_words_never_echo_the_word(self):
        file = self.file("README.md", "SYNTHETIC_PRIVATE_MARKER")
        findings = release.scan_paths(self.root, [file], ["SYNTHETIC_PRIVATE_MARKER"])
        self.assertEqual(findings[0]["category"], "private_deny_word")
        self.assertNotIn("SYNTHETIC_PRIVATE_MARKER", str(findings))

    def test_archive_timestamps_and_hashes_are_reproducible(self):
        with mock.patch.dict(sys.modules, {"check_release": release}):
            export_spec = importlib.util.spec_from_file_location("source_exporter", ROOT / "scripts/export_release.py")
            exporter = importlib.util.module_from_spec(export_spec)
            export_spec.loader.exec_module(exporter)
        file = self.file("README.md", "Authored synthetic source")
        with (
            mock.patch.object(exporter, "ROOT", self.root),
            mock.patch.object(exporter, "scan", return_value=([file], [])),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(exporter.main(), 0)
            archive = self.root / "dist/chatlens-0.1.0-source.zip"
            first = hashlib.sha256(archive.read_bytes()).digest()
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.namelist(), ["chatlens/README.md", "chatlens/SOURCE_MANIFEST.json"])
                self.assertTrue(all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in handle.infolist()))
            self.assertEqual(exporter.main(), 0)
            self.assertEqual(hashlib.sha256(archive.read_bytes()).digest(), first)

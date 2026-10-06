"""Tests for the plugin version registry and updater."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plugins import check_registry, plugin_versions, set_plugin_version


class PluginUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "ailake-core").mkdir()
        (self.root / "ailake-core" / "Cargo.toml").write_text('version = "1.2.3"\n')
        (self.root / "demo").mkdir()
        self.file = self.root / "demo" / "build.gradle.kts"
        self.file.write_text('version = "1.2.3"\n')
        self.plugin = {
            "id": "demo",
            "directory": "demo",
            "targets": [{
                "file": "demo/build.gradle.kts",
                "pattern": r'(?m)^version = "[^"]+"$',
                "replacement": 'version = "{version}"',
                "version_pattern": r'(?m)^version = "([^"]+)"$',
            }],
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_updates_single_target_and_reports_path(self):
        changed = set_plugin_version(self.root, self.plugin, "1.2.4")
        self.assertEqual(changed, [self.file])
        self.assertEqual(plugin_versions(self.root, self.plugin), [("demo/build.gradle.kts", "1.2.4")])

    def test_check_registry_detects_version_drift(self):
        self.file.write_text('version = "1.2.4"\n')
        errors = check_registry(self.root, {"plugins": [self.plugin]})
        self.assertEqual(len(errors), 1)
        self.assertIn("differs from core 1.2.3", errors[0])

    def test_update_fails_if_target_is_ambiguous(self):
        self.file.write_text('version = "1.2.3"\nversion = "1.2.3"\n')
        with self.assertRaisesRegex(ValueError, "expected one version declaration, found 2"):
            set_plugin_version(self.root, self.plugin, "1.2.4")

    def test_update_does_not_rewrite_an_aligned_target(self):
        changed = set_plugin_version(self.root, self.plugin, "1.2.3")
        self.assertEqual(changed, [])
        self.assertEqual(self.file.read_text(), 'version = "1.2.3"\n')

    def test_registry_file_is_valid_json(self):
        registry_path = Path(__file__).with_name("plugins.json")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(registry["plugins"]), 1)


if __name__ == "__main__":
    unittest.main()

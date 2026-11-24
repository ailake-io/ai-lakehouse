import json
import tempfile
import unittest
from pathlib import Path

from scripts.release_tag import validate_versions


class ReleaseTagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('core', 'python', 'scripts', 'fuzz'):
            (self.root / name).mkdir()
        (self.root / 'Cargo.toml').write_text('[workspace]\nmembers = ["core", "python"]\n')
        (self.root / 'core/Cargo.toml').write_text('[package]\nname = "core"\nversion = "1.2.3"\n')
        (self.root / 'python/Cargo.toml').write_text('[package]\nname = "python"\nversion = "0.1.0"\n')
        (self.root / 'scripts/plugins.json').write_text(json.dumps({'plugins': [{
            'version_policy': 'independent', 'targets': [{'file': 'python/Cargo.toml'}]
        }]}))
        for name in ('Cargo.lock', 'fuzz/Cargo.lock'):
            (self.root / name).write_text('[[package]]\nname = "core"\nversion = "1.2.3"\n')
        (self.root / 'CHANGELOG.md').write_text('## [Unreleased]\n\n## [1.2.3] — 2026-11-23\n')

    def test_prepared_release_accepts_independent_package_versions(self):
        self.assertEqual(validate_versions(self.root, 'v1.2.3'), '1.2.3')

    def test_rejects_nonstable_or_noncore_tags(self):
        for tag in ('1.2.3', 'python-v1.2.3', 'v1.2.3-rc1', 'v01.2.3', 'v1.2.3.4'):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_versions(self.root, tag)

    def test_rejects_tag_that_does_not_match_manifests(self):
        with self.assertRaisesRegex(ValueError, 'version must be 1.2.4'):
            validate_versions(self.root, 'v1.2.4')

    def test_rejects_each_stale_lockfile(self):
        for name in ('Cargo.lock', 'fuzz/Cargo.lock'):
            path = self.root / name
            original = path.read_text()
            path.write_text(original.replace('1.2.3', '1.2.2'))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'stale version'):
                validate_versions(self.root, 'v1.2.3')
            path.write_text(original)

    def test_requires_release_notes_to_be_prepared(self):
        (self.root / 'CHANGELOG.md').write_text('## [Unreleased]\n')
        with self.assertRaisesRegex(ValueError, 'dated'):
            validate_versions(self.root, 'v1.2.3')


if __name__ == '__main__':
    unittest.main()

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.python_release import git, resolve


class PythonReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('ailake-core', 'ailake-py'):
            (self.root / name).mkdir()
        self.core = self.root / 'ailake-core/Cargo.toml'
        self.python = self.root / 'ailake-py/Cargo.toml'
        self.core.write_text('[package]\nversion = "1.2.3"\n')
        self.python.write_text('[package]\nversion = "0.1.0"\n')
        git(self.root, 'init', '-b', 'main')
        git(self.root, 'config', 'user.name', 'Release tests')
        git(self.root, 'config', 'user.email', 'release-tests@example.invalid')
        self.commit()
        git(self.root, 'tag', 'v1.2.3')
        git(self.root, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
        self.core_sha = git(self.root, 'rev-parse', 'HEAD')

    def commit(self):
        git(self.root, 'add', '.')
        git(self.root, 'commit', '-m', 'Release fixture')

    def test_manual_release_uses_core_tag_instead_of_newer_main(self):
        (self.root / 'later-change.txt').write_text('after the core release')
        self.commit()
        result = resolve(self.root, '', 'refs/heads/main')
        self.assertEqual(result, {'version': '1.2.3', 'tag': 'python-v1.2.3',
                                 'source_sha': self.core_sha, 'core_tag': 'v1.2.3'})

    def test_reusable_call_accepts_matching_core_tag(self):
        self.assertEqual(resolve(self.root, '1.2.3', 'refs/tags/v1.2.3')['source_sha'],
                         self.core_sha)

    def test_rejects_unpublished_version_or_invalid_version(self):
        for version in ('1.2.4', '1.2.3-rc1', '01.2.3', 'v1.2.3'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                resolve(self.root, version, 'refs/heads/main')

    def test_rejects_other_branch_or_wrong_tag(self):
        for ref in ('refs/heads/develop', 'refs/tags/v1.2.4'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                resolve(self.root, '1.2.3', ref)

    def test_rejects_core_tag_version_drift(self):
        git(self.root, 'tag', 'v1.2.4')
        with self.assertRaisesRegex(ValueError, 'Core manifest'):
            resolve(self.root, '1.2.4', 'refs/heads/main')

    def test_reuses_existing_matching_python_tag(self):
        self.python.write_text('[package]\nversion = "1.2.3"\n')
        self.commit()
        git(self.root, 'tag', 'python-v1.2.3')
        result = resolve(self.root, '1.2.3', 'refs/heads/main')
        self.assertEqual(result['source_sha'], git(self.root, 'rev-parse', 'HEAD'))

    def test_rejects_existing_python_tag_version_drift(self):
        git(self.root, 'tag', 'python-v1.2.3')
        with self.assertRaisesRegex(ValueError, 'Existing python-v1.2.3'):
            resolve(self.root, '1.2.3', 'refs/heads/main')

    def test_rejects_core_tag_outside_main_history(self):
        self.core.write_text('[package]\nversion = "1.2.4"\n')
        self.commit()
        git(self.root, 'tag', 'v1.2.4')
        with self.assertRaises(subprocess.CalledProcessError):
            resolve(self.root, '1.2.4', 'refs/heads/main')


if __name__ == '__main__':
    unittest.main()

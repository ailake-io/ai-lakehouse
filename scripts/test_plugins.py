"""Tests for the plugin version registry and updater."""

import json
import tempfile
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.plugins import check_registry, plugin_versions, set_plugin_version, load_registry, bump_core_cargo_versions


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

    def test_all_documentation_references_are_checked_and_updated(self):
        self.file.write_text('release-v1.2.3 and release-v1.2.4')
        self.plugin['targets'] = [{
            'file': 'demo/build.gradle.kts', 'pattern': r'(release-v)[0-9.]+',
            'replacement': r'\g<1>{version}', 'version_pattern': r'release-v([0-9.]+)',
            'multiple': True,
        }]
        self.assertTrue(check_registry(self.root, {'plugins': [self.plugin]}))
        set_plugin_version(self.root, self.plugin, '1.2.3')
        self.assertEqual(self.file.read_text(), 'release-v1.2.3 and release-v1.2.3')
        self.assertEqual(check_registry(self.root, {'plugins': [self.plugin]}), [])

    def test_core_prepare_bumps_python_and_its_install_examples(self):
        py = self.root / 'demo' / 'Cargo.toml'
        py.write_text('[package]\nversion = "1.2.3"\n')
        self.file.write_text('ailake==1.2.3')
        self.plugin['version_policy'] = 'core'
        self.plugin['targets'] = [
            {'file':'demo/Cargo.toml', 'pattern':r'(?m)^version = "[^"]+"$',
             'replacement':'version = "{version}"', 'version_pattern':r'(?m)^version = "([^"]+)"$'},
            {'file':'demo/build.gradle.kts', 'pattern':r'(ailake==)[0-9.]+',
             'replacement':r'\g<1>{version}', 'version_pattern':r'ailake==([0-9.]+)'},
        ]
        registry = {'plugins':[self.plugin]}
        bump_core_cargo_versions(self.root,'1.2.3','1.2.4',registry)
        set_plugin_version(self.root,self.plugin,'1.2.4')
        self.assertEqual(check_registry(self.root,registry),[])
        self.assertIn('1.2.4',py.read_text())
        self.assertEqual(self.file.read_text(),'ailake==1.2.4')

    def test_duckdb_docs_keep_the_engine_abi_version(self):
        plugin = next(p for p in load_registry()['plugins'] if p['id'] == 'duckdb')
        target = next(dict(t) for t in plugin['targets'] if t['file'].endswith('.md'))
        target['file'] = 'demo/build.gradle.kts'
        self.file.write_text('duckdb-v0.1.17/ailake-duckdb-v2.0.0.duckdb_extension')
        set_plugin_version(self.root,{'targets':[target]},'0.1.18')
        self.assertEqual(self.file.read_text(),'duckdb-v0.1.18/ailake-duckdb-v2.0.0.duckdb_extension')

    def test_actual_core_preparation_keeps_plugins_independent(self):
        source = Path(__file__).resolve().parents[1]
        registry = load_registry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {'scripts/plugins.py', 'scripts/plugins.json', 'Cargo.toml', 'ailake-core/Cargo.toml'}
            for plugin in registry['plugins']:
                paths.update(target['file'] for target in plugin['targets'])
            paths.update(target['file'] for target in registry.get('core_targets', []))
            for path in paths:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / path, destination)
            before = {p['id']: plugin_versions(root, p) for p in registry['plugins']
                      if p.get('version_policy') == 'independent'}
            current = plugin_versions(root, next(p for p in registry['plugins'] if p['id']=='python'))[0][1]
            major, minor, patch = map(int, current.split('.'))
            next_version = f'{major}.{minor}.{patch+1}'
            for args in (
                ['bump-core-cargo', '--from-version', current, '--version', next_version],
                ['update', '--policy', 'core', '--version', next_version],
                ['check'],
            ):
                subprocess.run([sys.executable, str(root/'scripts/plugins.py'), *args],
                               cwd=root, check=True, capture_output=True, text=True)
            for plugin in registry['plugins']:
                after = plugin_versions(root, plugin)
                if plugin['id'] in before:
                    self.assertEqual(after, before[plugin['id']])
                else:
                    self.assertTrue(all(version == next_version for _, version in after))

    def test_registry_file_is_valid_json(self):
        registry_path = Path(__file__).with_name("plugins.json")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(registry["plugins"]), 1)


if __name__ == "__main__":
    unittest.main()
